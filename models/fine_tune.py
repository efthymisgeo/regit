import os
import sys
import csv
import json
import torch
#import pandas as pd
import numpy as np
#import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torchvision.models as torch_models
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from modules.models import CNN2D, CNNFC
from utils.model_utils import train, test, validate, EarlyStopping
from utils.mnist import MNIST, CIFAR10
from utils.config_loader import print_config
from utils.opts import load_experiment_options
from models.end2end import run_training
from captum.attr import LayerConductance
from utils.importance import Importance, Attributor

def set_parameter_requires_grad(model, requires_grad=False):
    """Sets requires_grad for all the vgg parameters in a model.
    Args:
        model(nn model): model to alter.
        requires_grad(bool): whether the model
            requires grad.
    """
    for param in model.features.parameters():
        param.requires_grad = requires_grad

def check_directory_and_create(dir_path, exists_warning=False):
    """
    Checks if the path specified is a directory or creates it if it doesn't
    exist.
    
    Args:
        dir_path (string): directory path to check/create
    
    Returns:
        (string): the input path
    """
    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            raise ValueError(f"Given path {dir_path} is not a directory")
        elif exists_warning:
            print(f"WARNING: Already existing experiment folder {dir_path}."
                  "It is recommended to change experiment_id in "
                  "configs/exp_config.json file. Proceeding by overwriting")
    else:
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def make_vgg_clf(old_params, clf_params, reinit=True):
    """construct a vgg classifier
    Args:
        old_params (torch.nn.Sequential): the module which consists of the
            pre-trained model parameters
        clf_params (dict): a dict which has all the necessary classifier info
        reinit (bool): handles reinitializing opiton. can be false only when
            resuming training in the same dataset (ImageNet)
    """
    fc_list = clf_params["fc_layers"]
    p_drop = clf_params["p_drop"] 
    fc = []
    # append linear layers
    for i_fc in range(0, len(fc_list)-1):
        fc.append(nn.Linear(fc_list[i_fc], fc_list[i_fc+1]))
        fc.append(nn.ReLU())
        fc.append(nn.Dropout(p_drop))
    fc.append(nn.Linear(fc_list[-2], fc_list[-1]))
    return nn.Sequential(*fc)

if __name__ == '__main__':
    exp_config = load_experiment_options()
    model_setup = exp_config["model_opt"]
    exp_setup = exp_config["exp_opt"]
    data_setup = exp_config["data_opt"]

    model = torch_models.vgg11(pretrained=True)
    print(model)

    set_parameter_requires_grad(model, requires_grad=False)

    # resnet18 = torch_models.resnet18(pretrained=True)
    # densenet = torch_models.densenet161(pretrained=True)
    # googlenet = torch_models.googlenet(pretrained=True)
    # shufflenet = torch_models.shufflenet_v2_x1_0(pretrained=True)
    # mobilenet = torch_models.mobilenet_v2(pretrained=True)
    # resnext50_32x4d = torch_models.resnext50_32x4d(pretrained=True)
    # wide_resnet50_2 = torch_models.wide_resnet50_2(pretrained=True)
    # mnasnet = torch_models.mnasnet1_0(pretrained=True)
    # model_list = [vgg11,
    #               resnet18,
    #               densenet,
    #               googlenet,
    #               shufflenet,
    #               mobilenet,
    #               resnext50_32x4d,
    #               wide_resnet50_2,
    #               mnasnet]
    # for mdl in model_list:
    #     print(f"{mdl}")
    #     import pdb; pdb.set_trace()

    # load cifar-10 for finetuning
    data = CIFAR10(data_setup, exp_setup)
    train_loader, val_loader = data.get_train_val_loaders()
    test_loader = data.get_test_loader()

    # get rid of last layers
    ft_method = exp_setup["ft_method"]
    if 'vgg' in exp_setup['model_name']:
        if ft_method == "dropout":
            new_clf = make_vgg_clf(model.classifier,
                                   model_setup["FC"],
                                   reinit=True)
        elif ft_method == "i-drop":
            pass
        elif ft_method == "plain":
            pass
        else:
            pass
        model.classifier = new_clf
    print(model)

    
    