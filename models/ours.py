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

import copy
import glob

class Ours(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(self.args, False)
        self.teacher_model_old = IncrementalNet(self.args, False)
        self.teacher_model_new = None

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


    def after_task(self):
        self._known_classes = self._total_classes
        self.teacher_model_old= copy.deepcopy(self._network)
        del self.teacher_model_new
    
    # def replace_fc(self,trainloader, model, args):
    #     model = model.eval()
    #     embedding_list = []
    #     label_list = []
    #     with torch.no_grad():
    #         for i, batch in enumerate(trainloader):
    #             (_,data,label) = batch
    #             data = data.cuda()
    #             label = label.cuda()
    #             embedding = model(data)["features"]
    #             embedding_list.append(embedding.cpu())
    #             label_list.append(label.cpu())
    #     embedding_list = torch.cat(embedding_list, dim=0)
    #     label_list = torch.cat(label_list, dim=0)

    #     class_list = np.unique(self.train_dataset.labels)
    #     proto_list = []
    #     for class_index in class_list:
    #         # print('Replacing...',class_index)
    #         data_index = (label_list == class_index).nonzero().squeeze(-1)
    #         embedding = embedding_list[data_index]
    #         proto = embedding.mean(0)
    #         self._network.fc.weight.data[class_index] = proto
    #     return model

    def teacher_train(self, train_loader, test_loader):
        epochs = self.args["teacher_epochs"]
        self.teacher_model_new.to(self._device)
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
                test_acc = self._compute_accuracy(self.teacher_model_new, test_loader)
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

        if len(self._multiple_gpus) > 1:
            self.teacher_model_new = self.teacher_model_new.module
        savepath = "pretrained/teacher_task{}_epoch{}.pth".format(self._cur_task, epochs)
        torch.save(self.teacher_model_new.state_dict(), savepath)
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # self._network = IncrementalNet(self.args, False) # 초기화
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


        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task > 0:
            self.teacher_model_new = IncrementalNet(self.args, False)
            self.teacher_model_new.update_fc(self._total_classes-self._known_classes)

            if self.args["use_pretrained"]:
                PATH = "pretrained/{}/teacher_task{}_epoch{}.pth".format(self.args["pretrained_dir"],self._cur_task,200)
                try:
                    print('Load', PATH)
                    self.teacher_model_new.load_state_dict(torch.load(PATH))
                    # if len(self._multiple_gpus) > 1:
                        # self.teacher_model_new = nn.DataParallel(self.teacher_model_new, self._multiple_gpus)
                except BaseException as e:
                    print(e)
                    print('Load failed train new teacher', PATH)
                    # if len(self._multiple_gpus) > 1:
                    #     self.teacher_model_new = nn.DataParallel(self.teacher_model_new, self._multiple_gpus)
                    self.teacher_train(self.train_loader, self.test_loader)

        self._train(self.train_loader, self.test_loader)

            # if self._cur_task > 0:
            #     self.teacher_model_new = self.teacher_model_new.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
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
            if self.args["use_pretrained"]:
                PATH = "pretrained/{}/teacher_task{}_epoch{}.pth".format(self.args["pretrained_dir"],self._cur_task,200)
                try:
                    print('Load', PATH)
                    self._network.load_state_dict(torch.load(PATH))
                except:
                    print('Load failed, train new student', PATH)
                    self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self.teacher_model_new.to(self._device)
            for layer in self._network.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.lrate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay
            )
            
            self._update_representation(train_loader, test_loader, optimizer, scheduler)





    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            # self.teacher_model.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                
                loss = F.cross_entropy(logits, targets)
                # teacher_logits = self.teacher_model(inputs)["logits"]
                # teacher_logits = Variable(teacher_logits, requires_grad=False)
                # loss = self.loss_kd_self(logits, targets, teacher_logits, self.args)

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

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        savepath = "pretrained/teacher_task{}_epoch{}.pth".format(self._cur_task, self.args["init_epoch"])
        torch.save(self._network.state_dict(), savepath)




    def loss_kd_self(self, outputs, labels, teacher_outputs):
        """
        loss function for self training: Tf-KD_{self}
        """
        T = self.args["temperature"]
        alpha = self.args["alpha"]
        D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T) * self.args["multiplier"]  # multiple is 1.0 in most of cases, some cases are 10 or 50
        KD_loss =  alpha*D_KL

        return KD_loss

    def perason_dist(self, Y_s, Y_t):
        inter = 0
        intra = 0
        batch_len = Y_t.shape[0]
        class_len = Y_t.shape[1]
        for batch in range(batch_len):
            u=Y_s[batch,:]
            v=Y_t[batch,:]
            inter += (1 - torch.corrcoef(torch.stack([u,v]))[0,1])
        for cls in range(class_len):
            u=Y_s[:,cls]
            v=Y_t[:,cls]
            intra += (1 - torch.corrcoef(torch.stack([u,v]))[0,1])
        inter = inter/batch_len
        intra = intra/class_len
        return inter, intra
    
    def cosine_similarity(self, a, b, eps=1e-8):
        
        return (a * b).sum(1) / (a.norm(dim=1)* b.norm(dim=1) + eps)


    def pearson_correlation(self, a, b, eps=1e-8):
        return self.cosine_similarity(a - a.mean(1).unsqueeze(1),
                                b - b.mean(1).unsqueeze(1), eps)


    def inter_class_relation(self, y_s, y_t, mode):
        #32341
        if mode == 0:
            return 1 - self.pearson_correlation(y_s, y_t).mean()
        else:
            return abs(self.pearson_correlation(y_s, y_t).mean())

    def intra_class_relation(self, y_s, y_t, mode):
        return self.inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1), mode)

    def loss_relation(self, u, v, mode=0):
        T = self.args["temperature"]
        U = F.softmax(u/T, dim=1)
        V = F.softmax(v/T, dim=1)
        inter_loss = T**2 * self.inter_class_relation(U, V, mode)
        intra_loss = T**2 * self.intra_class_relation(U, V, mode)
        return inter_loss, intra_loss

    
    
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        codebook = torch.nn.Parameter(torch.randn(64,64)).to(self._device)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self.teacher_model_old.eval()
            self.teacher_model_new.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                student_feat = self._network(inputs)["features"]
                old_teacher_feat = self.teacher_model_old(inputs)["features"]
                new_teacher_feat = self.teacher_model_new(inputs)["features"]
                fake_targets = targets - self._known_classes
                a = old_teacher_feat @ codebook
                b = new_teacher_feat @ codebook
                c = student_feat @ codebook
                loss_d1, loss_d2 = self.loss_relation(a,b, mode=1)

                old_teacher_logits = self.teacher_model_old.fc(a)["logits"]
                new_teacher_logits = self.teacher_model_new.fc(b)["logits"]
                if len(self._multiple_gpus) > 1:
                    logits_com = self._network.module.fc(c)["logits"]
                else:
                    logits_com = self._network.fc(c)["logits"]
                old_logits = logits_com[:, :self._known_classes]
                new_logits = logits_com[:, self._known_classes:]

                loss_cls = F.cross_entropy(new_logits, fake_targets)
                # loss_cls = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                loss_inter_old, loss_intra_old = self.loss_relation(old_logits, old_teacher_logits)
                loss_inter_new, loss_intra_new = self.loss_relation(new_logits, new_teacher_logits)
                loss_inter, loss_intra = self.loss_relation(logits, torch.cat([old_teacher_logits, new_teacher_logits], dim=1))

                # loss = loss_d1 + loss_d2 + loss_cls + loss_inter_old+ loss_intra_old+ loss_inter_new + loss_intra_new+loss_inter+loss_intra
                loss = loss_cls

                print("{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|".format(loss_d1 , loss_d2 , loss_cls , loss_inter_old , loss_intra_old , loss_inter_new ,  loss_intra_new , loss_inter , loss_intra))

                # teacher_logits = self.teacher_model_new(inputs)["logits"]
                # teacher_logits = Variable(teacher_logits, requires_grad=False)
                # loss_clf = self.loss_kd_self(logits[:, self._known_classes :], fake_targets, teacher_logits)
                # loss = loss_cls + 0*loss_inter_old + 0*loss_intra_old + loss_inter_new + loss_intra_new

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 1 == 0: # edited
                test_acc = self._compute_accuracy(self._network, test_loader)
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