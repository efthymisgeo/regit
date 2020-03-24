import os
import sys
import copy
import json
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


def plot_dict(ax, t, x, id, line, title="loss curves"):
    """plots a list of values
    """
    #fig, ax = plt.subplots()
    plt.xticks(t, t)
    #ax.set_title(title)
    for k, v in x.items():
        print(len(v))
        ax.plot(t, v, line, label=id+"_"+k)
    ax.legend()


if __name__ == "__main__":
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #plt.rc('text', usetex=True)
    path1 = "experiments/CIFAR10/_plain_cifar_0.01/CNN2D__plain_cifar_0.01_run_0.json"
    path2 = "experiments/CIFAR10/_dropout_0.5_0.01/CNN2D__dropout_0.5_0.01_run_0.json"
    path3 = "experiments/CIFAR10/_plain_cifar_0.01/CNN2D__plain_cifar_0.01_run_0.json"
    paths = [(path1, "no-drop", "-") , (path2, "drop", "--")]
    
    path1 = "experiments/CIFAR10/_plain_cifar_0.01/CNN2D__plain_cifar_0.01_run_0.json"
    path2 = "experiments/CIFAR10/_dropout_0.5_0.01/CNN2D__dropout_0.5_0.01_run_0.json"
    path3 = "experiments/CIFAR10/_plain_cifar_0.01/CNN2D__plain_cifar_0.01_run_0.json"
    paths = [(path1, "no-drop", "-") , (path2, "drop", "--")]
    mpl.style.use("ggplot")

    fig, ax = plt.subplots()
    for pth, nm, line in paths:
        with open (pth, "r") as fd:
            summary = json.load(fd)

        losses = summary["loss"]
        accuracies = summary["acc"]
        best_epoch = summary["best_epoch"]
        test_acc = summary["test_acc"]
        # hack to work
        # should be removed in the future
        losses["test"].append(losses["test"][-1])
        t = np.arange(0, len(losses["train"]))
        plot_dict(ax, t, losses, nm, line)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')    
    plt.savefig('losses.png', bbox_inches='tight')
