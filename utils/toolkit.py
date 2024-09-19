import os
import numpy as np
import torch
import  json
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def accuracy_per_class(y_pred, y_true):
    assert len(y_pred) == len(y_true), "Data length error."
    acc_per_class = {}

    # per_class (by datamanager)
    for class_id in range(0, np.max(y_true) + 1, 1):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + 1)
        )[0]
        label = "{}".format(
            str(class_id).rjust(2, "0")
        )
        acc_per_class[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    return acc_per_class

def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true) + 1, increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1: 
        fc_weight = model._network.fc.weight.data    
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")

def save_model(args, model):
    #used in PODNet
    _path = os.path.join(args['logfilename'], "model.pt")
    if len(args['device']) > 1:
        weight = model._network   
    else:
        weight = model._network.cpu()
    torch.save(weight, _path)


def draw_cnn(data, xlabel, ylabel, x_prefix='class_', y_prefix='task_', fname='result'):

    f, axs = plt.subplots(len(ylabel), 1, gridspec_kw={'hspace': 0}, sharex=True,figsize=(len(xlabel),len(ylabel)))
    for i in range(len(ylabel)):
        hh = np.expand_dims(data[i],0)
        mask = np.logical_not(hh>=0)
        yl = y_prefix + str(ylabel[i])
        xl = [x_prefix+x for x in xlabel]
        ax=sns.heatmap(hh, annot=True,vmin=0, vmax=100, linewidth=.5,fmt=".0f", cmap="RdYlGn", square=True,  cbar=False, ax=axs[i], center=hh[0,i], xticklabels=xl, yticklabels=[yl],mask=mask)

        ax.tick_params(axis='both', which='both', length=3)
        # ax.set(xlabel="Episode", ylabel="")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # ax.tick_params(top=True, labeltop=True,bottom=False, labelbottom=False)
    figure = ax.get_figure()    
    figure.savefig(fname+'.png', dpi=800, format='png')
    figure.savefig(fname+'.pdf', dpi=800, format='pdf')
    plt.cla()
    plt.clf()
    plt.close()

def draw_cls(data, xlabel, ylabel, x_prefix='class_', y_prefix='task_', fname='result'):

    f, axs = plt.subplots(1, len(xlabel), gridspec_kw={'hspace': 0, 'wspace':0}, sharey=True,figsize=(len(xlabel),len(ylabel)))
    for i in range(len(xlabel)):
        hh = np.expand_dims(data[:,i],1)
        for j in range(hh.shape[0]):
            if hh[j,0]>0:
                fwd_idx=j
                break
        mask = np.logical_not(hh>0)
        xl = x_prefix + str(xlabel[i])
        yl = [y_prefix+y for y in ylabel] 
        ax=sns.heatmap(hh, annot=True,vmin=0, vmax=100, linewidth=.5,fmt=".0f", cmap="RdYlGn", square=True,  cbar=False, center=hh[fwd_idx,0], ax=axs[i], xticklabels=[xl], yticklabels=yl,mask=mask)

        ax.set(xlabel="", ylabel="")
        if i>0:
            ax.tick_params(axis='y', which='both', length=0)
        else:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.tick_params(top=True, labeltop=True,bottom=False, labelbottom=False)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
    figure = ax.get_figure()    
    figure.savefig(fname+'.png', dpi=800, format='png')
    figure.savefig(fname+'.pdf', dpi=800, format='pdf')
    plt.cla()
    plt.clf()
    plt.close()


def t_sne_train(features1_epoch, labels1, features2_epoch, labels2, fname):
    from sklearn.manifold import TSNE
    plt.style.use(['seaborn-paper'])
    tsne = TSNE(n_components=2, random_state=0)
    cat = features1_epoch+features2_epoch
    labels = labels1 + labels2

    cluster = np.array(tsne.fit_transform(np.array(cat)))
    labels = np.array(labels)
    # labels2 = np.array(labels2)
    plt.figure(figsize=(10, 10))
    cifar = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    order = [4, 2, 7, 6, 0, 3, 5, 8, 9, 1]
    # order = [i - 1 for i in order]
    cifar = [cifar[i] for i in order]

    for i, label in zip(range(10), cifar):
        idx = np.where(labels == i)
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)

    plt.legend()
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    plt.close()



def t_sne_test(features1_epoch, labels_epoch, fname):
    from sklearn.manifold import TSNE
    plt.style.use(['seaborn-paper'])
    tsne = TSNE(n_components=2, random_state=0)
    cluster = np.array(tsne.fit_transform(np.array(features1_epoch)))
    labels_epoch = np.array(labels_epoch)
    plt.figure(figsize=(10, 10))
    cifar = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    order = [4, 2, 7, 6, 0, 3, 5, 8, 9, 1]
    # order = [i - 1 for i in order]
    cifar = [cifar[i] for i in order]

    for i, label in zip(range(10), cifar):
        idx = np.where(labels_epoch == i)
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)
    
    plt.legend()
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    plt.close()
