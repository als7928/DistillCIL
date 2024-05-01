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
from torch.nn.functional import adaptive_max_pool2d, adaptive_avg_pool2d

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
    def __init__(self, temperature, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_logits, teacher_logits, targets=None, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_logits / self.temperature, dim=1),
                                    torch.softmax(teacher_logits / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss

        hard_loss = self.cross_entropy_loss(student_logits, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss
    
class DISTLoss(nn.Module):
    """
    A loss module for Knowledge Distillation from A Stronger Teacher (DIST).
    Referred to https://github.com/hunto/image_classification_sota/blob/main/lib/models/losses/dist_kd.py

    Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu: `"Knowledge Distillation from A Stronger Teacher" <https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html>`_ @ NeurIPS 2022 (2022)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :param beta: balancing factor for inter-loss.
    :type beta: float
    :param gamma: balancing factor for intra-loss.
    :type gamma: float
    :param tau: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type tau: float
    """
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0, eps=1e-8, **kwargs):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = eps

    @staticmethod
    def cosine_similarity(a, b, eps=1e-8):
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)
    
    def pearson_correlation(self,y_s, y_t, eps):
        return self.cosine_similarity(y_s - y_s.mean(1).unsqueeze(1), y_t - y_t.mean(1).unsqueeze(1), eps=eps)

    def inter_class_relation(self, y_s, y_t):
        return 1 - self.pearson_correlation(y_s, y_t, self.eps).mean()

    def intra_class_relation(self, y_s, y_t):
        return self.inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

    def forward(self, student_logits, teacher_logits, *args, **kwargs):
        y_s = (student_logits / self.tau).softmax(dim=1)
        y_t = (teacher_logits / self.tau).softmax(dim=1)
        inter_loss = self.tau ** 2 * self.inter_class_relation(y_s, y_t)
        intra_loss = self.tau ** 2 * self.intra_class_relation(y_s, y_t)
        return self.beta * inter_loss , self.gamma * intra_loss
    
class PKTLoss(nn.Module):
    """
    A loss module for probabilistic knowledge transfer (PKT). Refactored https://github.com/passalis/probabilistic_kt/blob/master/nn/pkt.py

    Nikolaos Passalis, Anastasios Tefas: `"Learning Deep Representations with Probabilistic Knowledge Transfer" <https://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.html>`_ @ ECCV 2018 (2018)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param eps: constant to avoid zero division.
    :type eps: float

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`PKTLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'PKTLoss'
          kwargs:
            student_module_path: 'fc'
            student_module_io: 'input'
            teacher_module_path: 'fc'
            teacher_module_io: 'input'
            eps: 0.0000001
    """
    def __init__(self, eps=0.0000001):
        super().__init__()
        self.eps = eps
    def cosine_similarity_loss(self, student_outputs, teacher_outputs):
        # Normalize each vector by its norm
        norm_s = torch.sqrt(torch.sum(student_outputs ** 2, dim=1, keepdim=True))
        student_outputs = student_outputs / (norm_s + self.eps)
        student_outputs[student_outputs != student_outputs] = 0

        norm_t = torch.sqrt(torch.sum(teacher_outputs ** 2, dim=1, keepdim=True))
        teacher_outputs = teacher_outputs / (norm_t + self.eps)
        teacher_outputs[teacher_outputs != teacher_outputs] = 0

        # Calculate the cosine similarity
        student_similarity = torch.mm(student_outputs, student_outputs.transpose(0, 1))
        teacher_similarity = torch.mm(teacher_outputs, teacher_outputs.transpose(0, 1))

        # Scale cosine similarity to 0..1
        student_similarity = (student_similarity + 1.0) / 2.0
        teacher_similarity = (teacher_similarity + 1.0) / 2.0

        # Transform them into probabilities
        student_similarity = student_similarity / torch.sum(student_similarity, dim=1, keepdim=True)
        teacher_similarity = teacher_similarity / torch.sum(teacher_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        return torch.mean(teacher_similarity *
                          torch.log((teacher_similarity + self.eps) / (student_similarity + self.eps)))

    def forward(self, student_penultimate_outputs, teacher_penultimate_outputs, *args, **kwargs):
        return self.cosine_similarity_loss(student_penultimate_outputs, teacher_penultimate_outputs)

class FSPLoss(nn.Module):
    """
    A loss module for the flow of solution procedure (FSP) matrix.

    Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim: `"A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning" <https://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017)

    :param fsp_pairs: configuration of teacher-student module pairs to compute the loss for the FSP matrix.
    :type fsp_pairs: dict

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`FSPLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'FSPLoss'
          kwargs:
            fsp_pairs:
              pair1:
                teacher_first:
                  io: 'input'
                  path: 'layer1'
                teacher_second:
                  io: 'output'
                  path: 'layer1'
                student_first:
                  io: 'input'
                  path: 'layer1'
                student_second:
                  io: 'output'
                  path: 'layer1'
                weight: 1
              pair2:
                teacher_first:
                  io: 'input'
                  path: 'layer2.1'
                teacher_second:
                  io: 'output'
                  path: 'layer2'
                student_first:
                  io: 'input'
                  path: 'layer2.1'
                student_second:
                  io: 'output'
                  path: 'layer2'
                weight: 1
    """
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def compute_fsp_matrix(first_feature_map, second_feature_map):
        first_h, first_w = first_feature_map.shape[2:4]
        second_h, second_w = second_feature_map.shape[2:4]
        target_h, target_w = min(first_h, second_h), min(first_w, second_w)
        if first_h > target_h or first_w > target_w:
            first_feature_map = adaptive_max_pool2d(first_feature_map, (target_h, target_w))

        if second_h > target_h or second_w > target_w:
            second_feature_map = adaptive_max_pool2d(second_feature_map, (target_h, target_w))

        first_feature_map = first_feature_map.flatten(2)
        second_feature_map = second_feature_map.flatten(2)
        hw = first_feature_map.shape[2]
        return torch.matmul(first_feature_map, second_feature_map.transpose(1, 2)) / hw

    def forward(self, student_fmaps, teacher_fmaps, *args, **kwargs):
        fsp_loss = 0
        student_first_feature_map = student_fmaps[0]
        student_second_feature_map = student_fmaps[1]
        student_fsp_matrices = self.compute_fsp_matrix(student_first_feature_map, student_second_feature_map)
        teacher_first_feature_map = teacher_fmaps[0]
        teacher_second_feature_map = teacher_fmaps[1]
        teacher_fsp_matrices = self.compute_fsp_matrix(teacher_first_feature_map, teacher_second_feature_map)
        factor = 1
        fsp_loss += factor * (student_fsp_matrices - teacher_fsp_matrices).norm(dim=1).sum()
        batch_size = student_first_feature_map.shape[0]

        return fsp_loss / batch_size

class HierarchicalContextLoss(nn.Module):
    """
    A loss module for knowledge review (KR) method. Referred to https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia: `"Distilling Knowledge via Knowledge Review" <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html>`_ @ CVPR 2021 (2021)

    :param student_module_path: student model's module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Student4KnowledgeReview`.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param reduction: ``reduction`` for MSELoss.
    :type reduction: str or None
    :param output_sizes: output sizes of adaptive_avg_pool2d.
    :type output_sizes: list[int] or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`HierarchicalContextLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Student4KnowledgeReview` for the student model.

        criterion:
          key: 'HierarchicalContextLoss'
          kwargs:
            student_module_path: 'abf_modules.4'
            student_module_io: 'output'
            teacher_module_path: 'layer1.-1.relu'
            teacher_module_io: 'input'
            reduction: 'mean'
            output_sizes: [4, 2, 1]
    """
    def __init__(self, reduction='mean', output_sizes=None, **kwargs):
        super().__init__()
        if output_sizes is None:
            output_sizes = [4, 2, 1]

        self.criteria = nn.MSELoss(reduction=reduction)
        self.output_sizes = output_sizes

    def forward(self, student_features, teacher_features, *args, **kwargs):
        _, _, h, _ = student_features.shape
        loss = self.criteria(student_features, teacher_features)
        weight = 1.0
        total_weight = 1.0
        for k in self.output_sizes:
            if k >= h:
                continue

            proc_student_features = adaptive_avg_pool2d(student_features, (k, k))
            proc_teacher_features = adaptive_avg_pool2d(teacher_features, (k, k))
            weight /= 2.0
            loss += weight * self.criteria(proc_student_features, proc_teacher_features)
            total_weight += weight
        return loss / total_weight
    

class SingleKD(BaseLearner):
    def __init__(self, args):
        super().__init__(args)


        self.args = args
        self.t_args =self.args
        self.t_args["convnet_type"] = self.args["teacher_convnet_type"]
        self._network = IncrementalNet(self.args, False)
        self.teacher_model = IncrementalNet(self.t_args, False)

        self.num_workers = self.args["num_workers"]
        self.batch_size = self.args["batch_size"]
        self.init_lr = self.args["init_lr"]
        self.init_milestones = self.args["init_milestones"]
        self.init_lr_decay = self.args["init_lr_decay"]
        self.init_weight_decay = self.args["init_weight_decay"]

        self.teacher_lrate = self.args["teacher_lrate"]
        self.teacher_lrate_decay = self.args["teacher_lrate_decay"]
        self.teacher_weight_decay = self.args["teacher_weight_decay"]
        self.teacher_milestones = self.args["teacher_milestones"]

    def after_task(self):
        self._known_classes = self._total_classes

    def teacher_train(self, train_loader, test_loader):
        epochs = self.args["teacher_epochs"]
        self.teacher_model.to(self._device)
        teacher_optimizer = optim.SGD(
            self.teacher_model.parameters(),
            lr=self.teacher_lrate,
            momentum=0.9,
            weight_decay=self.teacher_weight_decay,
        )  # 1e-5
        teacher_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=teacher_optimizer, milestones=self.teacher_milestones, gamma=self.teacher_lrate_decay
        )

        logging.info(
            "Teacher model: learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.teacher_model.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self.teacher_model(inputs)["logits"]

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
                test_acc = self._compute_accuracy(self.teacher_model, test_loader)
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
            self.teacher_model = self.teacher_model.module
        savepath = "pretrained/teacher_task{}_{}_tstacc_{:.2f}.pth".format(self._cur_task, self.t_args["convnet_type"], test_acc)
        torch.save(self.teacher_model.state_dict(), savepath)
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task > 0:
            return

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


        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self.teacher_model = IncrementalNet(self.args, False)
        self.teacher_model.update_fc(self._total_classes-self._known_classes)

        if self.args["use_pretrained"]:
            PATH = self.args["pretrained_dir"]
            try:
                print('Load', PATH)
                self.teacher_model.load_state_dict(torch.load(PATH))
                if len(self._multiple_gpus) > 1:
                    self.teacher_model = nn.DataParallel(self.teacher_model, self._multiple_gpus)
            except BaseException as e:
                print(e)
                print('Load failed', PATH, 'train new teacher')
                if len(self._multiple_gpus) > 1:
                    self.teacher_model = nn.DataParallel(self.teacher_model, self._multiple_gpus)
                self.teacher_train(self.train_loader, self.test_loader)

        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self.teacher_model.to(self._device)

        optimizer = optim.SGD(
            self._network.parameters(),
            momentum=0.9,
            lr=self.init_lr,
            weight_decay=self.init_weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay
        )

        self._init_train(train_loader, test_loader, optimizer, scheduler)


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self.teacher_model.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out_student = self._network(inputs)
                out_teacher = self.teacher_model(inputs)
                logits_student = out_student["logits"]
                logits_teacher = out_teacher["logits"]
                fmaps_student =out_student["fmaps"]
                fmaps_teacher = out_teacher["fmaps"]
                features_student  = out_student["features"]
                features_teacher  = out_teacher["features"]

                
                loss_cls = F.cross_entropy(logits_student, targets)
                
                if self.args["mode"] == "Vanila":
                    loss_func = KDLoss(temperature=self.args["temperature"], alpha=None)
                    alpha = self.args["alpha"]
                    loss_kd = loss_func(logits_student, logits_teacher)
                    loss = alpha*loss_cls + (1-alpha)*loss_kd
                    if i == len(train_loader)-1:
                        print("epoch {}: {:.3f}|{:.3f}|{:.3f}|".format(epoch, loss, alpha*loss_cls, (1-alpha)*loss_kd))
                elif self.args["mode"] == "DIST":
                    loss_func = DISTLoss(tau=self.args["temperature"])
                    loss_inter, loss_intra = loss_func(logits_student, logits_teacher)
                    loss = loss_inter + loss_intra
                    if i == len(train_loader)-1:
                        print("epoch {}: {:.3f}|{:.3f}|{:.3f}|{:.3f}|".format(epoch, loss, loss_cls, loss_inter, loss_intra))
                elif self.args["mode"] == "PKT":
                    loss_func = PKTLoss()
                    loss_pkt = loss_func(logits_student, logits_teacher)
                    loss =  loss_pkt
                    if i == len(train_loader)-1:
                        print("epoch {}: {:.3f}|{:.3f}|{:.3f}|".format(epoch, loss, loss_cls, loss_pkt))
                # elif self.args["mode"] == "FSP":
                #     loss_func = FSPLoss()
                #     loss_fsp = loss_func(fmaps_student, fmaps_teacher)
                #     loss = loss_fsp
                #     print("epoch {}: {:.3f}|{:.3f}|{:.3f}|".format(epoch, loss, loss_cls, loss_fsp))
                # elif self.args["mode"] == "HC":
                #     loss_func = HierarchicalContextLoss()
                #     loss_hc = loss_func(features_student, features_teacher)
                #     loss = loss_hc
                #     print("epoch {}: {:.3f}|{:.3f}|{:.3f}|".format(epoch, loss, loss_cls, loss_hc))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits_student, dim=1)
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
        savepath = "pretrained/student_task{}_{}_tstacc_{:.2f}.pth".format(self._cur_task, self.args["convnet_type"],test_acc)
        torch.save(self._network.state_dict(), savepath)
