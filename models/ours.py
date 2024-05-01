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

    def forward(self, student_logits, teacher_logits, targets=None, *args, **kwargs):
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
            train_total_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self._train(self.train_loader, self.test_loader)
        if hasattr(self._network, 'module'):
            self._network = self._network.module
        if hasattr(self.teacher_model_new, 'module'):
            self.teacher_model_new = self.teacher_model_new.module
        if hasattr(self.teacher_model_old, 'module'):
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
            self._update_representation(self.total_train_loader, test_loader, optimizer, scheduler)


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
        codebook = torch.nn.Parameter(torch.randn(64,64)).to(self._device)
        kd_loss = KDLoss(temperature=self.args["temperature"])
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self.teacher_model_old.eval()
            self.teacher_model_new.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fake_targets = targets - self._known_classes

                student_logits = self._network(inputs)["logits"]
                student_feat = self._network(inputs)["features"]
                old_teacher_feat = self.teacher_model_old(inputs)["features"]
                new_teacher_feat = self.teacher_model_new(inputs)["features"]
                a = old_teacher_feat @ codebook
                b = new_teacher_feat @ codebook
                c = student_feat @ codebook

                old_teacher_logits = self.teacher_model_old.fc(a)["logits"]
                if len(self._multiple_gpus) > 1:
                    new_teacher_logits = self.teacher_model_new.module.fc(b)["logits"]
                    logits_com = self._network.module.fc(c)["logits"]
                else:
                    logits_com = self._network.fc(c)["logits"]
                    new_teacher_logits = self.new_teacher_logits.fc(c)["logits"]

                student_logits_old = logits_com[:, :self._known_classes]
                student_logits_new = logits_com[:, self._known_classes:]

                loss_cls = F.cross_entropy(student_logits, fake_targets)
                
                loss_kd_old = kd_loss(student_logits_old, old_teacher_logits)
                loss_kd_new = kd_loss(student_logits_new, new_teacher_logits)

                loss = loss_cls + loss_kd_old + loss_kd_new

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
                
                print("{:.3f}|{:.3f}|{:.3f}|{:.3f}|".format(loss, loss_cls, loss_kd_old, loss_kd_new))
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