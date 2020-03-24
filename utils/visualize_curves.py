import os
import sys
import copy
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))



REF_DICT_NAMES = {
    "loss":{"train":[], "test":[], "val":[]},
    "acc":{"train":[], "test":[], "val":[]},
    "best_epoch":0,
    "p_drop":0.0,
    "switches":[],
    "test_acc":0.0
}


def parse_args():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--mdl_config", required=True,
    #                     help="Path to model configuration file"),
    # parser.add_argument("-e", "--exp_config", required=False,
    #                     default="configs/exp_config.json",
    #                     help="Path to experiment configuration file."
    #                          "Handles optimizer, lr, batch size, etc")
    # parser.add_argument("-d", "--dt_config", required=False,
    #                     default="configs/mnist_config.json",
    #                     help="Path to data cnfiguration fil
    # TODO
    pass


def plot_dict(ax, t, x, id, line, title="loss curves"):
    """
    Plots a list of values
    Args:
        ax:
        t (list): list of ints which relate to the x-axis
        x (dict): {"train":[], "val":[], "test":[]}
        id (str): plot identifier string e.g smart-drop
        line (str): line identifier
        title (str):
        skip_val (bool): flag to skip validation set values
    """
    #fig, ax = plt.subplots()
    #plt.xticks(t, t, rotation=30)
    #ax.set_title(title)
    for k, v in x.items():
        if k == "val" and skip_val: continue
        ax.plot(t, v, line, label=id+"_"+k)
    ax.legend()


if __name__ == "__main__":
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #plt.rc('text', usetex=True)
    accuracies = True
    if accuracies:
        figname = "Accuracy (%)"
        save_as = "acc.png"
    else:
        figname = "Loss"
        save_as = "loss.png"
   
    path1 = "experiments/CIFAR10/_plain_cifar_0.01/CNN2D__plain_cifar_0.01_run_0.json"
    path2 = "experiments/CIFAR10/_dropout_0.5_0.01/CNN2D__dropout_0.5_0.01_run_0.json"
    path3 = "experiments/CIFAR10/_smart_0.001_0.00004/CNN2D__smart_0.001_0.00004_run_0.json"

    
    # paths = [(path1, "no-drop", "-") ,
    #          (path2, "drop", "--"),
    #          (path3, "drop", "-*")]
    
    skip_val = True

    # path1 = "experiments/CIFAR10/_plain_cifar_0.01/CNN2D__plain_cifar_0.01_run_0.json"
    # path2 = "experiments/CIFAR10/_dropout_0.5_0.01/CNN2D__dropout_0.5_0.01_run_0.json"
    # path3 = "experiments/CIFAR10/_smart_0.001_0.00004/CNN2D__smart_0.001_0.00004_run_0.json"
    paths = [(path1, "no-drop", "-"),
             (path2, "drop", "--"),
             (path3, "smart-drop", "-.")]
    mpl.style.use("ggplot")

    fig, ax = plt.subplots()
    plt.yscale('linear')
    plt.xscale('linear')
    connectionstyle='angle, angleA=-120, angleB=180, rad=10'
    for pth, nm, line in paths:
        with open (pth, "r") as fd:
            summary = json.load(fd)

        losses = summary["loss"]
        accuracies = summary["acc"]
        best_epoch = summary["best_epoch"]
        test_acc = summary["test_acc"]
        # hack to work
        # should be removed in the future
        if nm != "smart-drop":
            accuracies["test"].append(accuracies["test"][-1])
            losses["test"].append(losses["test"][-1])
        #t = np.arange(0, len(losses["train"]))
        if accuracies:
            t = np.arange(0, len(accuracies["train"]))
            ax.annotate(str(test_acc),
                        xy=(best_epoch, test_acc),
                        xytext=(best_epoch - 4, 15 + test_acc),
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle=connectionstyle,
                                        facecolor='white',
                                        shrink=0.05))
            plot_dict(ax, t, accuracies, nm, line, skip_val)
        else:
            t = np.arange(0, len(losses["train"]))    
            plot_dict(ax, t, losses, nm, line, skip_val)
    plt.xlabel('Epochs')
    plt.ylabel(figname)
    plt.savefig(save_as, bbox_inches='tight')
    
