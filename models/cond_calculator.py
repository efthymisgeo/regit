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
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.importance import Importance, Attributor
import matplotlib.font_manager


# sns.set(font="Times New Roman",
#         rc={
#  "axes.axisbelow": False,
#  "axes.edgecolor": "lightgrey",
#  "axes.facecolor": "None",
#  "axes.grid": False,
#  "axes.labelcolor": "dimgrey",
#  "axes.spines.right": False,
#  "axes.spines.top": False,
#  "figure.facecolor": "white",
#  "lines.solid_capstyle": "round",
#  "patch.edgecolor": "w",
#  "patch.force_edgecolor": True,
#  "text.color": "dimgrey",
#  "xtick.bottom": False,
#  "xtick.color": "dimgrey",
#  "xtick.direction": "out",
#  "xtick.top": False,
#  "ytick.color": "dimgrey",
#  "ytick.direction": "out",
#  "ytick.left": False,
#  "ytick.right": False})

# sns.set_context("notebook", rc={"font.size":10,
#                                 "axes.titlesize":10,
#                                 "axes.labelsize":18})

sns.set(font="Franklin Gothic Book",
        rc={
 "axes.axisbelow": False,
 "axes.edgecolor": "lightgrey",
 "axes.facecolor": "None",
 "axes.grid": False,
 "axes.labelcolor": "dimgrey",
 "axes.spines.right": False,
 "axes.spines.top": False,
 "figure.facecolor": "white",
 "lines.solid_capstyle": "round",
 "patch.edgecolor": "w",
 "patch.force_edgecolor": True,
 "text.color": "dimgrey",
 "xtick.bottom": False,
 "xtick.color": "dimgrey",
 "xtick.direction": "out",
 "xtick.top": False,
 "ytick.color": "dimgrey",
 "ytick.direction": "out",
 "ytick.left": False,
 "ytick.right": False})
sns.set_context("notebook", rc={"font.size":12,
                                "axes.titlesize":12,
                                "axes.labelsize":14})



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


ATTRIBUTE_SETUP = {
    "sample_batch": None,
    "n_steps": 15
}

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

COLOR_LIST = [
    "green",
    CB91_Amber,
    CB91_Violet,
    ]


def load_model(saved_model, checkpoint_path):
    
    # load model
    saved_model.load_state_dict(torch.load(checkpoint_path,
                                           map_location='cpu'))

    model = Toy_FC()
    model.fc[0] = saved_model.fc[0]
    model.fc[-1] = saved_model.fc[1]
    return model

def attribute_loop(model,
                   data_loader,
                   attributor,
                   n_steps,
                   sample_batch,
                   sigma_attr,
                   sigma_input,
                   adapt_to_tensor,
                   momentum,
                   aggregate,
                   per_sample_noise,
                   respect_attr,
                   device):
    mean_conductance = torch.zeros(1024).to(device)
    rankings = None
    model.eval()
    for batch_idx, (data, target) in tqdm(enumerate(data_loader),
                                                    total=len(data_loader)):
        #import pdb; pdb.set_trace()
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        # print(target.shape)
        batch_size = data.size(0)
        baseline = 0.0 * data
        rankings, statistics, total_conductance, per_class_cond = \
            attributor.get_attributions(data,
                                     baselines=baseline,
                                     target=target,
                                     n_steps=n_steps,
                                     sample_batch=sample_batch,
                                     sigma_attr=sigma_attr,
                                     sigma_input=sigma_input,
                                     adapt_to_tensor=adapt_to_tensor,
                                     momentum=momentum,
                                     aggregate=aggregate,
                                     per_sample_noise=per_sample_noise,
                                     respect_attr=respect_attr,
                                     batch_idx=batch_idx,
                                     calc_stats=False)
        mean_conductance = mean_conductance + rankings[0]
        del rankings
        del total_conductance
        torch.cuda.empty_cache()

    return mean_conductance / batch_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model_path1", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("-m2", "--model_path2", required=False,
                        help="Path to model checkpoint")
    parser.add_argument("-m3", "--model_path3", required=False,
                        help="Path to model checkpoint")
    # parser.add_argument("-e", "--exp_config", required=False,
    #                     default="configs/exp_config.json",
    #                     help="Path to experiment configuration file."
    #                          "Handles optimizer, lr, batch size, etc")
    # parser.add_argument("-d", "--dt_config", required=False,
    #                     default="configs/mnist_config.json",
    #                     help="Path to data cnfiguration file")
    # load checkpoint
    args = parser.parse_args()
    # checkpoint_path = args.model_path
    # # load configs
    data_setup = load_config("configs/dataset/mnist.json")
    model_setup = load_config("configs/model/mnist_toy.json")
    exp_setup = load_config("configs/experiment/mnist_toy.json")
    exp_setup["kwargs"] = get_kwargs(exp_setup)
    exp_setup["device"] = get_device(exp_setup)
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
    saved_model_1 = CNNFC(input_shape=input_shape,
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
    
    saved_model_2 = CNNFC(input_shape=input_shape,
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

    saved_model_3 = CNNFC(input_shape=input_shape,
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



    print(args.model_path1)
    print(args.model_path2)
    print(args.model_path3)
    model_1 = load_model(saved_model_1, args.model_path1)
    model_2 = load_model(saved_model_2, args.model_path2)
    model_3 = load_model(saved_model_3, args.model_path3)
    model_1.to(device)
    model_2.to(device)
    model_3.to(device)
    
    # mean conductance calculation
    y_attr_1 = Attributor(model_1, [model_1.fc[0]])
    y_attr_2 = Attributor(model_2, [model_2.fc[0]])
    y_attr_3 = Attributor(model_3, [model_3.fc[0]])

    # _, test_acc_1, _, _ = test(model_1, test_loader, criterion, device)
    # _, test_acc_2, _, _ = test(model_2, test_loader, criterion, device)
    # _, test_acc_3, _, _ = test(model_3, test_loader, criterion, device)
    # print(f"The accuracy of the loaded checkpoint is {test_acc_1} and {test_acc_2} and {test_acc_3}")
    aggregate = True
    sample_batch = ATTRIBUTE_SETUP.get("sample_batch", None)
    n_steps = ATTRIBUTE_SETUP.get("n_steps", 5)
    sigma_attr = ATTRIBUTE_SETUP.get("sigma_attr", None)
    sigma_input = ATTRIBUTE_SETUP.get("sigma_input", None)
    momentum = ATTRIBUTE_SETUP.get("momentum", None)
    adapt_to_tensor = ATTRIBUTE_SETUP.get("adapt_to_tensor", False)
    per_sample_noise = ATTRIBUTE_SETUP.get("per_sample_noise", False)
    respect_attr = ATTRIBUTE_SETUP.get("respect_attr", False)
    # calculate mean of mean conductance
    data_loader = test_loader
    mean_cond_1 = attribute_loop(model_1,
                                 data_loader,
                                 y_attr_1,
                                n_steps,
                                sample_batch,
                                sigma_attr,
                                sigma_input,
                                adapt_to_tensor,
                                momentum,
                                aggregate,
                                per_sample_noise,
                                respect_attr,
                                device)
    # del model_1
    # torch.cuda.empty_cache()
    # model_2.to("cuda:1")
    mean_cond_2 = attribute_loop(model_2,
                                    data_loader,
                                    y_attr_2,
                                    n_steps,
                                    sample_batch,
                                    sigma_attr,
                                    sigma_input,
                                    adapt_to_tensor,
                                    momentum,
                                    aggregate,
                                    per_sample_noise,
                                    respect_attr,
                                    device)
    # del model_2
    # torch.cuda.empty_cache()
    # model_3.to("cuda:2")
    mean_cond_3 = attribute_loop(model_3,
                                    data_loader,
                                    y_attr_3,
                                    n_steps,
                                    sample_batch,
                                    sigma_attr,
                                    sigma_input,
                                    adapt_to_tensor,
                                    momentum,
                                    aggregate,
                                    per_sample_noise,
                                    respect_attr,
                                    device)
    # del model_3
    # torch.cuda.empty_cache()

    mean_cond_1 = mean_cond_1.detach().cpu().numpy()
    mean_cond_2 = mean_cond_2.detach().cpu().numpy()
    mean_cond_3 = mean_cond_3.detach().cpu().numpy()
    sorted_mean_cond_1 = np.sort(mean_cond_1)
    sorted_mean_cond_2 = np.sort(mean_cond_2)
    sorted_mean_cond_3 = np.sort(mean_cond_3)
    
    # important units first
    sorted_mean_cond_1 = np.flipud(sorted_mean_cond_1)
    sorted_mean_cond_2 = np.flipud(sorted_mean_cond_2)
    sorted_mean_cond_3 = np.flipud(sorted_mean_cond_3)


    
    # PLOTING
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOR_LIST)
    

    x_axis = np.arange(1024)
    fig, ax = plt.subplots()
    # plt.xscale('log')
    columns = ['Y-Drop', 'Drop', 'Plain']
    plt.plot(x_axis, sorted_mean_cond_1)
    plt.plot(x_axis, sorted_mean_cond_2)
    plt.plot(x_axis, sorted_mean_cond_3)
    ax.fill_between(x_axis, y1=sorted_mean_cond_1, y2=sorted_mean_cond_3[-1],
                    label=columns[0], alpha=0.2, color=COLOR_LIST[0])
    ax.fill_between(x_axis, y1=sorted_mean_cond_2, y2=sorted_mean_cond_3[-1],
                    label=columns[1], alpha=0.5, color=COLOR_LIST[1])
    ax.fill_between(x_axis, y1=sorted_mean_cond_3, y2=sorted_mean_cond_3[-1],
                    label=columns[2], alpha=0.4, color=COLOR_LIST[2])

    ax.set_xlabel("Neuron")
    ax.set_ylabel("Conductance")
    plt.legend(loc='upper right')
    plt.legend(frameon=False)
    # sns.set("rc"={"axes.grid": False})
    # plt.rc('grid', linestyle="--", color='darkgrey')
    # plt.grid(True)
    plt.savefig("mean_conductance_test-set.png",  bbox_inches = "tight")
    plt.savefig("mean_conductance_test-set.pdf", bbox_inches = "tight")
    plt.close()

    
    # cumsum 
    cumsum_1 = np.cumsum(sorted_mean_cond_1)
    cumsum_2 = np.cumsum(sorted_mean_cond_2)
    cumsum_3 = np.cumsum(sorted_mean_cond_3)


    text_at = [200, 400, 600, 800]
    tot_csum_1 = cumsum_1[-1]
    tot_csum_2 = cumsum_2[-1]
    tot_csum_3 = cumsum_3[-1]

    print(f"Final cumsum of Y-drop is {tot_csum_1}\n of Dropout is  {tot_csum_2} \n and of Plain is {tot_csum_3}.")


    fig, ax = plt.subplots()
    plt.plot(x_axis, cumsum_1, label=f"Y-Drop")
    plt.plot(x_axis, cumsum_2, label=f"Drop")
    plt.plot(x_axis, cumsum_3, label=f"Plain")
    plt.legend(loc='upper left')
    plt.legend(frameon=False)

    plt.scatter(text_at,
                cumsum_1[text_at],
                marker="8",
                color='tab:green',
                s=30,
                label='CumSum')
    plt.scatter(text_at,
                cumsum_2[text_at],
                marker="X",
                color="orange",
                s=30, 
                label='Troughs')
    plt.scatter(text_at,
                cumsum_3[text_at],
                marker="s",
                color="tab:purple",
                s=30, 
                label='xaxaxa')
    for t, p in zip(text_at, cumsum_1):
        # if t == 200:
        #     plt.text(t - 10,
        #         cumsum_3[t] + 1.5,
        #         "{0:.1f}%".format(cumsum_3[t]/tot_csum_3*100),
        #         horizontalalignment='left',
        #         color=COLOR_LIST[2])
        # elif t == 400:
        #     plt.text(t + 20,
        #         cumsum_3[t] -2 ,
        #         "{0:.1f}%".format(cumsum_3[t]/tot_csum_3*100),
        #         horizontalalignment='left',
        #         color=COLOR_LIST[2])
        # elif t == 600:
        #     plt.text(t - 2,
        #         cumsum_3[t] + 1.0,
        #         "{0:.1f}%".format(cumsum_3[t]/tot_csum_3*100),
        #         horizontalalignment='left',
        #         color=COLOR_LIST[2])
        # else:
        #     plt.text(t - 2,
        #         cumsum_3[t] + 1.5,
        #         "{0:.1f}%".format(cumsum_3[t]/tot_csum_3*100),
        #         horizontalalignment='left',
        #         color=COLOR_LIST[2])
        plt.text(t,
                cumsum_1[t] + 4,
                "{0:.1f}%".format(cumsum_1[t]/tot_csum_1*100),
                horizontalalignment='center',
                color="green")
        plt.text(t,
                cumsum_1[t] + 2,
                "{0:.1f}%".format(cumsum_2[t]/tot_csum_2*100),
                horizontalalignment='center',
                color="orange")
        plt.text(t,
                cumsum_1[t] + 3,
                "{0:.1f}%".format(cumsum_3[t]/tot_csum_3*100),
                horizontalalignment='center',
                color=COLOR_LIST[2])
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Cummulative Conductance Sum")
    plt.savefig("cumsum_mean_cond_test-set.png", bbox_inches = "tight")
    plt.savefig("cumsum_mean_cond_test-set.pdf", bbox_inches = "tight")
    plt.close()


    cumsum_1 = np.cumsum(np.flipud(sorted_mean_cond_1))
    cumsum_2 = np.cumsum(np.flipud(sorted_mean_cond_2))
    cumsum_3 = np.cumsum(np.flipud(sorted_mean_cond_3))


        # plt.text(df.date[t], df.traffic[t]-35, df.date[t],
        #         horizontalalignment='center', color='darkred')


    fig, ax = plt.subplots()
    plt.plot(x_axis, cumsum_1, label=f"Y-Drop")
    plt.plot(x_axis, cumsum_2, label=f"Drop")
    plt.plot(x_axis, cumsum_3, label=f"Plain")
    plt.legend(loc='upper left')
    plt.legend(frameon=False)
    plt.savefig("fliped-cumsum_mean_cond_test-set.png")
    plt.close()









