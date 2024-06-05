import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, draw_cnn, draw_cls
import os
import numpy as np
import time

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):


    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
 

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    if args["init_cls"] >= 5:
        cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    else:
        cnn_curve, nme_curve = {"top1": []}, {"top1": []}

    cnn_matrix, nme_matrix = [], []
    cls_matrix = []
    train_duration, eval_duration = [], []

    tm = time.localtime(time.time())
    starttime = time.strftime("%m%d_%H%M%S",tm)
    logging.info("Current time: {}".format(starttime))
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        
        start = time.perf_counter()
        model.incremental_train(data_manager)
        end = time.perf_counter()
        logging.info("Training duration: {}".format(end-start))
        train_duration.append(end-start)
        
        start = time.perf_counter()
        cnn_accy, nme_accy = model.eval_task()
        end = time.perf_counter()
        logging.info("Eval duration: {}".format(end-start))
        eval_duration.append(end-start)

        model.after_task()
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))
            logging.info("CLS: {}".format(cnn_accy["per_class"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)

            cls_keys = [key for key in cnn_accy["per_class"].keys()]
            cls_keys_sorted = sorted(cls_keys)
            cls_values = [cnn_accy["per_class"][key] for key in cls_keys_sorted]
            cls_matrix.append(cls_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            nme_curve["top1"].append(nme_accy["top1"])
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))

            if args["init_cls"] >= 5:
                cnn_curve["top5"].append(cnn_accy["top5"])
                nme_curve["top5"].append(nme_accy["top5"])

                logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
                logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("CLS: {}".format(cnn_accy["per_class"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            cls_keys = [key for key in cnn_accy["per_class"].keys()]
            cls_keys_sorted = sorted(cls_keys)
            cls_values = [cnn_accy["per_class"][key] for key in cls_keys_sorted]
            cls_matrix.append(cls_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            
            if args["init_cls"] >= 5:
                cnn_curve["top5"].append(cnn_accy["top5"])
                logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))


            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))


        if len(cls_matrix)>0:
            # np_acctable = np.zeros([task + 1, task + 1])
            np_acctable = np.zeros([data_manager.nb_tasks, args["init_cls"] + args["increment"]*(data_manager.nb_tasks-1)])
            for idxx, line in enumerate(cls_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            print('Class Accuracy Matrix (CLS):')
            print(np_acctable)
            tlabel = [str(i) for i in range(data_manager.nb_tasks)]
            draw_cls(np_acctable, data_manager._class_order, tlabel, y_prefix="episode_", x_prefix="class_",fname=logfilename+"_"+starttime+"_cls")

    if len(cnn_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        print(np_acctable)
        print('Forgetting (CNN):', forgetting)
        draw_cnn(np_acctable, tlabel, tlabel, y_prefix="task_", x_prefix="episode_",fname=logfilename+"_"+starttime+"_cnn")
        logging.info('Forgetting (CNN): {}'.format(forgetting))

            
    if len(nme_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (NME):')
        print(np_acctable)
        print('Forgetting (NME):', forgetting)
        logging.info('Forgetting (NME): {}'.format(forgetting))

    logging.info("{Total training duration}".format(sum(train_duration)))
    logging.info("{Total eval duration}".format(sum(eval_duration)))

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
