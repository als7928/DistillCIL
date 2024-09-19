from importlib.metadata import requires
import logging
from networkx import radius
import numpy as np
from sympy import O
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from utils.inc_net import IncrementalNet, BaseNet, OurNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy, t_sne_train, t_sne_test
from torch.autograd import Variable
from torch.autograd.function import Function
import os
import math
from convs.linears import SimpleLinear
import copy
import torchvision.transforms as transforms
import glob
from torch.nn import PairwiseDistance

milestones = [60, 120, 170]

class DecoderDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label_data = self.labels[idx]
        return input_data, label_data

    
class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, **kwargs):
        super(Encoder, self).__init__()
        self.lin = nn.Linear(input_dim, 64, **kwargs)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(64) 
        self.lin2 = nn.Linear(64, out_dim, **kwargs)

    def forward(self, x):
        z = self.lin(x)
        # z = self.tanh(z)
        z = self.relu(z)
        z = self.bn(z)
        z = self.lin2(z)
        return z

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pairwise_distance = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, negative_labels):
        
        pos_distance = self.pairwise_distance(anchor, positive)
        neg_distance = self.pairwise_distance(anchor, negative)

        # # Calculate the distance between negative samples of the same class
        # same_class_mask = (negative_labels.unsqueeze(1) == negative_labels.unsqueeze(0)).float()
        # neg_pairwise_distances = self.pairwise_distance(negative.unsqueeze(1), negative.unsqueeze(0))
        # neg_cluster_loss = torch.mean(neg_pairwise_distances * same_class_mask) 
 
        loss_contrastive = torch.mean(torch.pow(pos_distance, 2) +
                                      torch.pow(torch.clamp(self.margin - neg_distance, min=0.0), 2))
        return loss_contrastive

class Contrastive(nn.Module):
    def __init__(self, args, margin=1.0, old_classes=2, new_classes=2, feat_dim=64, device=None):
        super().__init__()
        self.args = args
        self.margin = margin
        self.device = device
        self.encoder_for_new = Encoder(self.args["enc_in_dim"], self.args["enc_in_dim"], bias=True).to(device)

        self.vec = nn.Parameter(torch.randn(old_classes, feat_dim), requires_grad=True)
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        self.fc = SimpleLinear(feat_dim, old_classes + new_classes).to(device)

    def forward(self, proto, new_feat, index, negative_labels, radius=None):
        proto_vec = self.vec[index].to(self.device)

        anchor = proto
        fill = torch.normal(0,1, size=anchor.shape, device=self.device)
        # fill = (torch.rand_like(anchor, device=self.device) - 0.5) * 2.0
        if self.args["ablation"] == "no_aug":
            positive = anchor
        elif radius is not None:
            positive = anchor + fill * radius
        else:
            positive = anchor + proto_vec.mul(fill)
        negative = self.encoder_for_new(new_feat)

        contrastive_loss = self.contrastive_loss(anchor, positive, negative, negative_labels) 
        return contrastive_loss, (anchor, positive, negative), (self.fc(positive)["logits"], self.fc(negative)["logits"])

class KDLoss(nn.KLDivLoss):
    """
    A standard knowledge distillation (KD) loss module.

    .. math::

       L_{KD} = \\alpha \cdot L_{CE} + (1 - \\alpha) \cdot \\tau^2 \cdot L_{KL}

    Geoffrey Hinton, Oriol Vinyals, Jeff Dean: `"Distilling the Knowledge in a Neural Network" <https://arxiv.org/abs/1503.02531>`_ @ NIPS 2014 Deep Learning and Representation Learning Workshop (2014)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param temperature: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type temperature: float
    :param alpha: balancing factor for :math:`L_{CE}`, cross-entropy.
    :type alpha: float
    :param beta: balancing factor (default: :math:`1 - \\alpha`) for :math:`L_{KL}`, KL divergence between class-probability distributions softened by :math:`\\tau`.
    :type beta: float or None
    :param reduction: ``reduction`` for KLDivLoss. If ``reduction`` = 'batchmean', CrossEntropyLoss's ``reduction`` will be 'mean'.
    :type reduction: str or None
    """
    def __init__(self, temperature, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_logits, teacher_logits, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_logits / self.temperature, dim=1),
                                    torch.softmax(teacher_logits / self.temperature, dim=1))
        return soft_loss

class Ours(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = OurNet(self.args, False)
        self.teacher_model_old = OurNet(self.args, False)
        self.teacher_model_new = OurNet(self.args, False)

        self.args = args
        self.num_workers = self.args["num_workers"]
        self.batch_size = self.args["batch_size"]
        self.epochs = self.args["epochs"]
        
        self.lrate = self.args["lrate"]
        self.weight_decay = self.args["weight_decay"]

        self._protos = []
        self._radius = 0
        self._radiuses = []

    @staticmethod
    def print_loss(*args):
        for l in args:
            print("{:.4f}".format(l), end ="|")
        print(" ")

    def save_weight(self, model, save_path):
        print('Saved pretrained weight', save_path)
        torch.save(model.state_dict(), save_path)

    def after_task(self):
        self._known_classes = self._total_classes
        # self.teacher_model_old= copy.deepcopy(self._network)
        self.teacher_model_old= self._network.copy().freeze()
    def _build_protos(self, data_manager):
            for class_idx in range(self._known_classes, self._total_classes):
                _, _, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                with torch.no_grad():
                    vectors, _ = self._extract_vectors(idx_loader)

                # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-8)).T
                # mean = np.mean(vectors, axis=0)
                # class_mean = mean / np.linalg.norm(mean)
                class_mean = np.mean(vectors, axis=0)

                self._protos.append(class_mean)
                cov = np.cov(vectors.T)
                self._radiuses.append(np.trace(cov)/vectors.shape[1]) 
            self._radius = np.sqrt(np.mean(self._radiuses))

    def teacher_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets - self._known_classes
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                
    def _teacher_train(self, train_loader, test_loader):
        epochs = self.args["teacher_epochs"]

        teacher_optimizer = optim.Adam(
            self.teacher_model_new.parameters(),
            lr=self.lrate,
            # momentum=0.9,
            weight_decay=self.weight_decay,
        )  # 1e-5
        # teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=teacher_optimizer, T_max=self.lrate, eta_min=0.0, last_epoch=-1)  
        teacher_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=teacher_optimizer, milestones=milestones, gamma=0.1)  
        logging.info(
            "Teacher model: learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.teacher_model_new.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self.teacher_model_new(inputs)["logits"]
                # logits = logits[:, self._known_classes :]
                targets = targets - self._known_classes
                
                loss = F.cross_entropy(logits, targets)

                teacher_optimizer.zero_grad()
                loss.backward()
                teacher_optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            teacher_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self.teacher_accuracy(self.teacher_model_new, test_loader)
                info = "Teacher: Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Teacher: Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
            logging.info(info)

        savepath = "pretrained/{}_{}_task{}-{}_hdim_{}.pth".format(self.args["dataset"], self.args["convnet_type"], self._known_classes, self._total_classes, self.args["enc_in_dim"])
        self.save_weight(self.teacher_model_new, savepath)


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        self._network.total_classes = self._total_classes
        self._network.known_classes = self._known_classes

        self._network.update_fc(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        teacher_test_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="test",  mode="test",
        )
        self.teacher_test_loader = DataLoader(
            teacher_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )      

        self._train(self.train_loader, self.test_loader)
        self._build_protos(data_manager)
        # self.build_rehearsal_memory(data_manager, 101)

    def _train(self, train_loader, test_loader):
        pre_train_loaded = False
        self._network.unfreeze()
        if self._cur_task == 0:
            optimizer = optim.Adam(
                self._network.parameters(),
                # momentum=0.9,
                lr=self.lrate,
                weight_decay=self.weight_decay
            )
            
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.lrate, eta_min=0.0, last_epoch=-1)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)  
            if self._cur_task < len(self.args["pretrained"]):
                PATH = os.path.join("pretrained/{}/{}".format(self.args["dataset"], self.args["convnet_type"]), self.args["pretrained"][self._cur_task])
                try:
                    print('Loading pretrained weight', PATH)
                    self._network.load_state_dict(torch.load(PATH))
                    pre_train_loaded = True
                    
                    self._network.to(self._device)
                except BaseException as e:
                    print(e)
            if not pre_train_loaded:
                
                self._network.to(self._device)
                self._init_train(train_loader, test_loader, optimizer, scheduler)

        else: # set teacher models for knowledge distillation
            self._network.to(self._device)
            self.teacher_model_new.update_fc(self._total_classes-self._known_classes)

            for layer in self.teacher_model_new.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            if self._cur_task < len(self.args["pretrained"]):
                PATH = os.path.join("pretrained/{}/{}".format(self.args["dataset"], self.args["convnet_type"]), self.args["pretrained"][self._cur_task])
                try:
                    print('Loading pretrained weight', PATH)
                    self.teacher_model_new.load_state_dict(torch.load(PATH))
                    
                    pre_train_loaded = True
                    
                    self.teacher_model_new.to(self._device)
                except BaseException as e:
                    print(e)
            if not pre_train_loaded:
                
                self.teacher_model_new.to(self._device)
                self._teacher_train(self.train_loader, self.teacher_test_loader)

            optimizer = optim.Adam(
                self._network.parameters(),
                lr=self.lrate,
                # momentum=0.9,
                weight_decay=self.weight_decay,
            )  # 1e-5

            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.lrate, eta_min=0.0, last_epoch=-1) 
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
             
            self._update_representation(self.train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

        savepath = "pretrained/{}_{}_task{}_epoch{}_tstacc_{:.2f}.pth".format(self.args["dataset"], self.args["convnet_type"], self._cur_task, epoch+1, test_acc)
        self.save_weight(self._network, savepath)

    def update_ema(self, student, old_teacher, alpha):
        for student_param, old_param in zip(student.named_parameters(), old_teacher.named_parameters()):
            student_name, student_data = student_param
            old_name, old_data = copy.deepcopy(old_param)
            if 'fc' in student_name and not 'convnet' in student_name:
                continue
                # student_data.data[:self._known_classes] = alpha * student_data.data[:self._known_classes] + (1 - alpha) * old_data.detach_().data
                # student_data.data[self._known_classes:] = alpha * student_data.data[self._known_classes:] + (1 - alpha) * new_data.detach_().data
            else:
                student_data.data = alpha * student_data.data + (1 - alpha) * old_data.detach_().data

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        contrastive_model = Contrastive(args=self.args, old_classes=self._known_classes, new_classes=self._total_classes-self._known_classes, feat_dim=self.args["enc_in_dim"], device=self._device)
        kd_loss = KDLoss(temperature=self.args["temperature"]).to(self._device)
        c_optimizer = optim.Adam(
            contrastive_model.parameters(),
            lr=self.lrate,
            # momentum=0.9,
            weight_decay=self.weight_decay,
        )  # 1e-5

        c_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=c_optimizer, milestones=milestones, gamma=0.1
        )
        
        # c_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=c_optimizer, T_max=self.lrate, eta_min=0.0, last_epoch=-1)  
        # self._network.learnable_tensor = nn.Parameter(torch.randn_like(inputs), requires_grad=True)
        # optimizer.add_param_group({'params': self._network.learnable_tensor})

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self.teacher_model_old.eval()
            self.teacher_model_new.eval()
            losses = 0.0
            correct, total = 0, 0

            st_feats = []
            positves = []
            negatives = []
            real = []
            proto_real = []

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # index = np.random.choice(range(self._known_classes),size=self.args["batch_size"]*int(self._known_classes/(self._total_classes-self._known_classes)),replace=True)

                c_optimizer.zero_grad()
                optimizer.zero_grad()
                index = np.random.choice(range(self._known_classes),size=self.args["batch_size"],replace=True)
                proto_features = np.array(self._protos)[index]
                proto_targets = index
                proto_features = torch.from_numpy(proto_features).float().to(self._device)
                proto_targets = torch.from_numpy(proto_targets).to(self._device)

                with torch.no_grad():
                    teacher_old_out  = self.teacher_model_old(inputs)
                    teacher_old_feat = teacher_old_out["features"]
                    teacher_new_out  = self.teacher_model_new(inputs)
                    teacher_new_feat = teacher_new_out["features"]

                if self.args["ablation"] == 'use_radius':
                    c_loss, com_space, com_logits = contrastive_model(proto_features, teacher_new_feat, index, targets, radius=self._radius)
                else: 
                    c_loss, com_space, com_logits = contrastive_model(proto_features, teacher_new_feat, index, targets)


                c_loss_a = F.cross_entropy(self.teacher_model_old.fc(com_space[1])["logits"], proto_targets)
                c_loss_b = F.cross_entropy(com_logits[0], proto_targets)
                c_loss_c = F.cross_entropy(com_logits[1], targets)
                if self.args["ablation"] == 'no_contrastive':
                    c_loss = 0
                closs = c_loss + c_loss_a + c_loss_b + c_loss_c

                closs.backward() 
                c_optimizer.step()

                c_optimizer.zero_grad()
                optimizer.zero_grad()
                
                _, com_space, com_logits = contrastive_model(proto_features, teacher_new_feat, index, targets)

                proto_aug = com_space[1]
                new_feat = com_space[2]
                
                # self.update_ema(self._network, self.teacher_model_old, 0.99)
                self._network.fc.weight.data = copy.deepcopy(contrastive_model.fc.weight.data) 
                self._network.fc.bias.data = copy.deepcopy(contrastive_model.fc.bias.data)  
                # Freeze the fc layer of self._network
                for param in self._network.fc.parameters():
                    param.requires_grad = False
                # self.update_ema(self._network, self.teacher_model_old, 0.90)
                # self._network.fc.weight = copy.deepcopy(contrastive_model.fc.weight)

                student_out = self._network(inputs)
                student_feat = student_out["features"]
                student_logits = student_out["logits"]

                # loss_new_cls = F.cross_entropy(student_logits, targets)
                lambd = 0.5
                loss_old = torch.dist(student_feat, teacher_old_feat, 2)
                loss_new = torch.dist(student_feat, new_feat, 2)

                # loss_old = kd_loss(student_feat, teacher_old_feat)
                # loss_new = kd_loss(student_feat, new_feat)/

                # loss_old = torch.mean(torch.abs(torch.sign(teacher_old_feat) - torch.abs(torch.sign(student_feat))))
                # loss_new = torch.mean(torch.abs(torch.sign(new_feat) - torch.abs(torch.sign(student_feat))))

                loss = (1-lambd) * loss_old + lambd * loss_new

                loss.backward() 
                optimizer.step()
                # self._network.fc.weight = copy.deepcopy(contrastive_model.fc.weight) 
                # self._network.fc.bias.data = copy.deepcopy(contrastive_model.fc.bias.data)  

                losses += loss.item()

                _, preds = torch.max(student_logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()

                total += len(targets)

            if self.args["drawing"] == True and epoch % 10 == 0:
                # anchors += com_space[0].detach().cpu().numpy().tolist()
                positves += proto_aug.detach().cpu().numpy().tolist()
                negatives += new_feat.detach().cpu().numpy().tolist()
                st_feats += student_feat.detach().cpu().numpy().tolist()
                # old_feats   += teacher_old_feat.cpu().numpy().tolist()
                real += targets.cpu().numpy().tolist()
                proto_real += proto_targets.cpu().numpy().tolist()
                # new_real += teacher_new_feat.cpu().numpy().tolist()

                print('plotting t-SNE ...')
                t_sne_train(positves, proto_real, negatives, real, 'change.png')
                # t_sne_test(anchors, proto_real, 'anchors.png')
                # t_sne_test(positves, proto_real, 'positves.png')
                # t_sne_test(negatives, real, 'negatives.png')
                t_sne_test(st_feats, real, 'student.png')
            scheduler.step()
            c_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 1 == 0: # edited
                test_acc = self._compute_accuracy(self._network, test_loader)
                test_nme = self._compute_nme(self._network, test_loader, self._protos/np.linalg.norm(self._protos,axis=1)[:,None])

                self.print_loss(closs, c_loss, c_loss_a, c_loss_b, c_loss_c)
                self.print_loss(loss, loss_old, loss_new)

                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Test_nme_old {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,  
                    test_nme['grouped']['old']
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)