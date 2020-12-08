from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os
import sys
import csv
from tqdm import tqdm
import json
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.models as torch_models
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from modules.vgg import *
from modules.models import CNN2D, CNNFC
from utils.model_utils import train, test, validate, EarlyStopping
from utils.mnist import MNIST, CIFAR10, CIFAR100, IMAGE_NET, STL10, SVHN, IMAGE_NET
from utils.config_loader import *
from utils.opts import load_experiment_options
import matplotlib.pyplot as plt
from models.end2end import run_training
from captum.attr import LayerConductance
import matplotlib.pyplot as plt
from utils.importance import Importance, Attributor


class Toy_FC(nn.Module):
    def __init__(self):
        super(Toy_FC, self).__init__()
        self.fc = nn.Sequential(
        nn.Linear(28*28, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 10)
    )

    def forward(self, x):
        x = x.view(-1, 28*28)
        out = self.fc(x)
        return out


def load_model(saved_model, checkpoint_path):
    
    # load model
    saved_model.load_state_dict(torch.load(checkpoint_path,
                                           map_location='cpu'))

    model = Toy_FC()
    model.fc[0] = saved_model.fc[0]
    model.fc[-1] = saved_model.fc[1]
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("-p", "--percentage", type=float, default=0.1,
                        help="percentage of pruned units")
    # parser.add_argument("-mask", "--masking", type=str,
    #                     choices=["random", "cond"],
    #                     default="random",
    #                     help="random or cond masking")
    # parser.add_argument("-p", "--percentage", type=float, default=0.1,
    #                     help="percentage of kept units")
    # parser.add_argument("-run", "--n_runs", type=int, default=10,
    #                     help="the number of inference runs")
    # parser.add_argument("--update_mask", action="store_true",
    #                     help="Flag which handles the update of masks in every"
    #                     "batch of the inference step. When False a unique mask"
    #                     "is appied")
    # parser.add_argument("--no_mask", action="store_true",
    #                     help="no masking is applied when this is called")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    # load data
    data_setup = load_config("configs/dataset/mnist.json")
    model_setup = load_config("configs/model/mnist_toy.json")
    exp_setup = load_config("configs/experiment/mnist_toy.json")
    exp_setup["kwargs"] = get_kwargs(exp_setup)
    exp_setup["device"] = args.device
    # train test split
    data = MNIST(data_setup, exp_setup)
    train_loader = data.get_train_loader()
    test_loader = data.get_test_loader()
    # preliminaries\
    device = exp_setup["device"]
    regularization = exp_setup["regularization"]
    fc_only = exp_setup.get("fc_only", False)
    importance = exp_setup["importance"]  # True to use importance
    use_drop_schedule = True if exp_setup["use_drop_schedule"] is not None else False  # True to use scheduler
    mixout = exp_setup["mixout"]
    plain_drop_flag = exp_setup["plain_drop"]
    if exp_setup["use_drop_schedule"] != {}: 
        custom_scheduler = exp_setup["use_drop_schedule"]["prob_scheduler"]
        if custom_scheduler == "Exp":
            gamma = exp_setup["use_drop_schedule"]["gamma"]
    
    if exp_setup["idrop"] != {}:
        map_rank_method = exp_setup["idrop"].get("method", "bucket")
        p_buckets = exp_setup["idrop"].get("p_buckets", [0.2, 0.8])
        inv_trick = exp_setup["idrop"].get("inv_trick", "dropout")
        alpha = exp_setup["idrop"].get("alpha", 1e-5)
        drop_low = exp_setup["idrop"].get("drop_low", True)
        sigma_drop = exp_setup["idrop"].get("sigma_drop", 0.05)
        rk_history = exp_setup["idrop"].get("rk_history", "short")
    else:
        map_rank_method = False
        p_buckets = [0.9, 0.1]
        inv_trick = "dropout"
        drop_low = True
        alpha = 1e-5
        sigma_drop = 0.05
        rk_history = "short"
    # loss function
    criterion = nn.CrossEntropyLoss()
    # model setup
    fc_setup = model_setup["FC"]
    input_shape = data_setup["input_shape"]
    
    # test_loss, test_acc, _, _ = test(model, test_loader, criterion, device)
    # print(f"The accuracy of the loaded checkpoint is {test_acc}")
    saved_model = CNNFC(input_shape=input_shape,
                    regularization=False,
                    activation=fc_setup["activation"],
                    fc_layers=fc_setup["fc_layers"],
                    add_dropout=fc_setup["fc_drop"],
                    p_drop=fc_setup["p_drop"],
                    drop_low=drop_low,
                    idrop_method=map_rank_method,
                    p_buckets=p_buckets,
                    inv_trick=inv_trick,
                    sigma_drop=sigma_drop,
                    alpha=alpha,
                    rk_history=rk_history,
                    pytorch_dropout=exp_setup["plain_drop"],
                    prior=exp_setup["prior"],
                    device=device,
                    fc_only=fc_only)
    
    model = load_model(saved_model, args.model_path)
    model.to(device)
    dense_layer = model.fc[0]
    #import pdb; pdb.set_trace()
    weight_pruner = prune.l1_unstructured(dense_layer, 'weight', amount=args.percentage)
    
    # bias_pruner = prune.l1_unstructured(dense_layer, 'bias', amount=args.percentage)
    # prune.remove(dense_layer, "weight")
    _, test_acc, _, _ = \
            test(model, test_loader, criterion, device)
    print(f"The accuracy of the loaded checkpoint is {test_acc}")





