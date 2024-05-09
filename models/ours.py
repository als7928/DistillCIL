import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, BaseNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from torch.autograd import Variable
import os

import copy
import glob

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        encoded = self.layer2(x)
        return encoded
    
class InterClassSeparationLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(InterClassSeparationLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.eps = 1e-8

    def forward(self, features, labels):
        # Initialize a centroid tensor
        centroids = torch.zeros((self.num_classes, self.feature_dim), device=features.device)
        count = torch.zeros(self.num_classes, device=features.device)

        # Accumulate features for each class
        for i in range(self.num_classes):
            mask = (labels == i)
            if torch.sum(mask) > 0:
                centroids[i] = torch.mean(features[mask], dim=0)
                count[i] = torch.sum(mask)

        # Calculate distance between centroids
        loss = 0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                if count[i] > 0 and count[j] > 0:
                    distance = torch.norm(centroids[i] - centroids[j]) / 16
                    loss = loss + torch.exp(-(distance + self.eps))  # Maximize separation
        return loss
 
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
        self._network = IncrementalNet(self.args, False)
        self.teacher_model_old = IncrementalNet(self.args, False)
        self.teacher_model_new = IncrementalNet(self.args, False)

        self.args = args
        self.num_workers = self.args["num_workers"]
        self.batch_size = self.args["batch_size"]
        self.epochs = self.args["epochs"]
        self.init_lr = self.args["init_lr"]
        self.init_milestones = self.args["init_milestones"]
        self.init_lr_decay = self.args["init_lr_decay"]
        self.init_weight_decay = self.args["init_weight_decay"]
        
        self.lrate_decay = self.args["lrate_decay"]
        self.lrate = self.args["lrate"]
        self.weight_decay = self.args["weight_decay"]
        self.milestones = self.args["milestones"]

    @staticmethod
    def print_loss(*args):
        for l in args:
            print("{:.4f}".format(l), end ="|")
        print(" ")

    def save_weight(self, model, save_path):
        if len(self._multiple_gpus) > 1 and isinstance(model, nn.DataParallel):
            model = model.module
        print('Saved pretrained weight', save_path)
        torch.save(model.state_dict(), save_path)
        if len(self._multiple_gpus) > 1:
            model = nn.DataParallel(model, self._multiple_gpus)

    def after_task(self):
        self._known_classes = self._total_classes
        # self.teacher_model_old= copy.deepcopy(self._network)
        self.teacher_model_old= self._network.copy().freeze()

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
        teacher_optimizer = optim.SGD(
            self.teacher_model_new.parameters(),
            lr=self.lrate,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )  # 1e-5
        teacher_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=teacher_optimizer, milestones=self.milestones, gamma=self.lrate_decay
        )

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

        savepath = "pretrained/{}_{}_task{}_epoch{}_tstacc_{:.2f}.pth".format(self.args["dataset"], self.args["convnet_type"], self._cur_task, epoch+1, test_acc)
        self.save_weight(self.teacher_model_new, savepath)

    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        for layer in self._network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self._network.update_fc(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
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

        train_total_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="train", mode="train"
        )
        self.total_train_loader = DataLoader(
            train_total_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        train_memory_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        class_sample_count = np.array([len(np.where(train_memory_dataset.labels==t)[0]) for t in np.unique(train_memory_dataset.labels)])
        weights = 1. / class_sample_count
        samples_weights = weights[train_memory_dataset.labels]
        assert len(samples_weights) == len(train_memory_dataset.labels)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
        
        self.train_memory_loader = DataLoader(
            train_memory_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=sampler
        )

        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.args["memory_per_class"])
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
        if isinstance(self.teacher_model_new, nn.DataParallel):
            self.teacher_model_new = self.teacher_model_new.module
        if isinstance(self.teacher_model_old, nn.DataParallel):
            self.teacher_model_old = self.teacher_model_old.module

    def _train(self, train_loader, test_loader):
        pre_train_loaded = False
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay
            )
            
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay
            )
            if self._cur_task < len(self.args["pretrained"]):
                PATH = os.path.join("pretrained/{}/{}".format(self.args["dataset"], self.args["convnet_type"]), self.args["pretrained"][self._cur_task])
                try:
                    print('Loading pretrained weight', PATH)
                    self._network.load_state_dict(torch.load(PATH))
                    pre_train_loaded = True
                    if len(self._multiple_gpus) > 1:
                        self._network = nn.DataParallel(self._network, self._multiple_gpus)
                    self._network.to(self._device)
                except BaseException as e:
                    print(e)
            if not pre_train_loaded:
                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
                self._network.to(self._device)
                self._init_train(train_loader, test_loader, optimizer, scheduler)

        else: # set teacher models for knowledge distillation
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
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
                    if len(self._multiple_gpus) > 1:
                        self.teacher_model_new = nn.DataParallel(self.teacher_model_new, self._multiple_gpus)
                    self.teacher_model_new.to(self._device)
                except BaseException as e:
                    print(e)
            if not pre_train_loaded:
                if len(self._multiple_gpus) > 1:
                    self.teacher_model_new = nn.DataParallel(self.teacher_model_new, self._multiple_gpus)
                self.teacher_model_new.to(self._device)
                self._teacher_train(self.train_loader, self.teacher_test_loader)

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.lrate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay
            )
            self._update_representation(self.train_memory_loader, test_loader, optimizer, scheduler)


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
   
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        codebook = torch.nn.Parameter(torch.randn(128,64)).to(self._device)
        # codebook = torch.nn.Linear(self._total_classes,self._total_classes,True).to(self._device)
        kd_loss = KDLoss(temperature=self.args["temperature"])
        myLoss2 = InterClassSeparationLoss(self._total_classes, feature_dim=64*2)
        encoder_for_student = Encoder(64, 64, 64).to(self._device)
        encoder_for_old = Encoder(64, 64, 64).to(self._device)
        encoder_for_new = Encoder(64, 64, 64).to(self._device)


        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self.teacher_model_old.eval()
            self.teacher_model_new.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = targets < self._known_classes # set True for old data
                # fake_targets = targets - self._known_classes

                student_feat = self._network(inputs)["features"]
                with torch.no_grad():
                    teacher_old_feat = self.teacher_model_old(inputs[mask])["features"]
                    teacher_new_feat = self.teacher_model_new(inputs[~mask])["features"]
                    teacher_old_logits = self.teacher_model_old.fc(teacher_old_feat)["logits"]


                com_old = encoder_for_old(teacher_old_feat) @ torch.t(encoder_for_student(student_feat))  # 63 128
                com_new = encoder_for_new(teacher_new_feat) @ torch.t(encoder_for_student(student_feat))
                com_student = torch.concat([com_old, com_new], dim=0) @ codebook
                breakpoint()

                # com_old = encoder_for_old(teacher_old_feat) @ codebook
                # com_new = encoder_for_new(teacher_new_feat) @ codebook
                # com_student = encoder_for_student(student_feat) @ codebook


                if len(self._multiple_gpus) > 1:
                    student_logits_com_old = self._network.module.fc(com_old)["logits"]
                    student_logits_com_new = self._network.module.fc(com_new)["logits"]
                    student_logits_com = self._network.module.fc(com_student)["logits"]
                    student_logits = self._network.module.fc(student_feat)["logits"] # [batch, total_classes]
                    with torch.no_grad():
                        teacher_new_logits = self.teacher_model_new.module.fc(teacher_new_feat)["logits"]
                else:
                    student_logits_com_old = self._network.fc(com_old)["logits"]
                    student_logits_com_new = self._network.fc(com_new)["logits"]
                    student_logits_com = self._network.fc(com_student)["logits"]
                    student_logits = self._network.fc(com_old)["logits"]
                    with torch.no_grad():
                        teacher_new_logits = self.teacher_model_new.fc(teacher_new_feat)["logits"]


                loss_1 = kd_loss(student_logits_com_old[:, :self._known_classes], teacher_old_logits)
                loss_2 = kd_loss(student_logits_com_new[:, self._known_classes:], teacher_new_logits)
                # loss_cls_com = F.cross_entropy(student_logits_com, targets)
                loss_3 = kd_loss(student_logits, student_logits_com)

                # loss_a_cls = F.cross_entropy(student_logits_com_a, targets)
                loss_cls = F.cross_entropy(student_logits, targets)

                # loss_kd = kd_loss(student_logits, student_logits_com_c)

                loss = loss_1 + loss_2 + loss_3 + loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(student_logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 1 == 0: # edited
                test_acc = self._compute_accuracy(self._network, test_loader)
                self.print_loss(loss, loss_1, loss_2, loss_3, loss_cls)

                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
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