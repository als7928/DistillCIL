import os
import random
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.custom_transforms import PermuteTensor
from torch.utils.data import Dataset

ROOT = './data'

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

############################################# Customized datasets
class RotatedMNIST(datasets.MNIST):
    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
            if True:#self.train:
                angle = random.normalvariate(0, 3.14/4)
                img = transforms.functional.rotate(img, angle * (180/3.14))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


#############################################
class iMNIST(iData):
    use_path = False
    train_trsf = [
        # transforms.Resize(32),
        # transforms.RandomCrop(28, padding=4),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Resize(32, antialias=True),
        transforms.Normalize(
            mean=(0.5,), std=(0.5,)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.MNIST(ROOT, train=True, download=True)
        test_dataset = datasets.MNIST(ROOT, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iMNIST28(iData):
    use_path = False
    train_trsf = [
        # transforms.Resize(32),
        # transforms.RandomCrop(28, padding=4),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        # transforms.Resize(32),
        transforms.Normalize(
            mean=(0.5,), std=(0.5,)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.MNIST(ROOT, train=True, download=True)
        test_dataset = datasets.MNIST(ROOT, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iFashionMNIST(iData):
    use_path = False
    train_trsf = [
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Resize(32),
        transforms.Normalize(
            mean=(0.5,), std=(0.5,)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.FashionMNIST(ROOT, train=True, download=True)
        test_dataset = datasets.FashionMNIST(ROOT, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iEMNIST(iData):
    use_path = False
    train_trsf = [
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Resize(32),
        transforms.Normalize(
            mean=(0.5,), std=(0.5,)
        ),
    ]

    class_order = np.arange(47).tolist()

    def download_data(self):
        train_dataset = datasets.EMNIST(ROOT, train=True, download=True, split="balanced")
        test_dataset = datasets.EMNIST(ROOT, train=False, download=True, split="balanced")
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iRotatedMNIST(iData):
    use_path = False
    train_trsf = [
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Resize(32),
        transforms.Normalize(
            mean=(0.5,), std=(0.5,)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = RotatedMNIST(ROOT, train=True, download=True)
        test_dataset = RotatedMNIST(ROOT, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iPermutedMNIST(iData):
    use_path = False
    train_trsf = [
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Resize(32),
        transforms.Normalize(
            mean=(0.5,), std=(0.5,)
        ),
        PermuteTensor((1,32,32)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.MNIST(ROOT, train=True, download=True)
        test_dataset = datasets.MNIST(ROOT, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class FastCelebA(Dataset):
    def __init__(self, data, attr):
        self.dataset = data
        self.attr = attr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.attr[index]
#############################################

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(ROOT, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(ROOT, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(ROOT, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(ROOT, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = os.path.join(ROOT, "[DATA-PATH]/train/")
        test_dir = os.path.join(ROOT, "[DATA-PATH]/val/")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = os.path.join(ROOT, "[DATA-PATH]/train/")
        test_dir = os.path.join(ROOT, "[DATA-PATH]/val/")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
