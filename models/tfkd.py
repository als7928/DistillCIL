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

class TFKD(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
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
            # self.teacher_model.eval()
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
        savepath = "pretrained/teacher_task{}_epoch{}.pth".format(self._cur_task, epochs)
        torch.save(self.teacher_model_new.state_dict(), savepath)
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
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
        if self._cur_task > 0:
            self.teacher_model_new = IncrementalNet(self.args, False)
            self.teacher_model_new.update_fc(self._total_classes-self._known_classes)
            if self.args["use_pretrained"]:
                PATH = glob.glob("pretrained/set_4276035891/teacher_task{}_epoch{}*".format(self._cur_task,200))[0]
                print('Load', PATH)
                self.teacher_model_new.load_state_dict(torch.load(PATH))

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            if self._cur_task > 0:
                self.teacher_model = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            if self._cur_task > 0:
                self.teacher_model = self.teacher_model.module

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
                PATH = glob.glob("pretrained/set_4276035891/student_task{}_epoch{}*".format(self._cur_task,200))[0]
                print('Load', PATH)
                self._network.load_state_dict(torch.load(PATH))
            else:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self.teacher_model_new.to(self._device)
            if not self.args["use_pretrained"]:
                self.teacher_train(train_loader, test_loader)
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


    def loss_kd_self(self, outputs, labels, teacher_outputs):
        """
        loss function for self training: Tf-KD_{self}
        """
        alpha = self.args["alpha"]
        T = self.args["temperature"]

        loss_CE = F.cross_entropy(outputs, labels)
        D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T) * self.args["multiplier"]  # multiple is 1.0 in most of cases, some cases are 10 or 50
        KD_loss =  (1. - alpha)*loss_CE + alpha*D_KL

        return KD_loss


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
        savepath = "pretrained/teacher_task{}_epoch{}.pth".format(self._cur_task, self.args["init_epoch"])
        torch.save(self._network.state_dict(), savepath)
        prog_bar.set_description(info)

        logging.info(info)
    
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                fake_targets = targets - self._known_classes
                
                teacher_logits = self.teacher_model_new(inputs)["logits"]
                teacher_logits = Variable(teacher_logits, requires_grad=False)
                loss_clf = self.loss_kd_self(logits[:, self._known_classes :], fake_targets, teacher_logits)
                loss = loss_clf

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