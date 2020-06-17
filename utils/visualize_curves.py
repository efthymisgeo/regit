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


def plot_dict(ax, t, x, id, line="-", title="loss curves", thresh=40):
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
        thresh (int): max epoch
    """
    #fig, ax = plt.subplots()
    #plt.xticks(t, t, rotation=30)
    #ax.set_title(title)
    for k, v in x.items():
        if (k == "val" and skip_val) or (k == 'test'):
            continue
        else:
            print(f"For name {id} and time horizon of {t}")
            kke = np.array(v[:thresh])
            ax.plot(t, kke, line, label=id+"_"+k)
    ax.legend()


if __name__ == "__main__":
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #plt.rc('text', usetex=True)
    path2exps = "experiments/CIFAR10/"
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--mdl_folder1", required=True,
                        help="Path to model experiment file")
    parser.add_argument("-m2", "--mdl_folder2", required=True,
                        help="Path to model experiment file")
    parser.add_argument("-m3", "--mdl_folder3", required=False,
                        help="Path to model experiment file")

    parser.add_argument("-n1", "--name_1", required=True,
                        help="Name of experiment 1")
    parser.add_argument("-n2", "--name_2", required=True,
                        help="Name of experiment 2")
    parser.add_argument("-n3", "--name_3", required=True,
                        help="Name of experiment 3")
    
    parser.add_argument("-m4", "--mdl_folder4", required=False,
                        help="Path to model experiment file")
    parser.add_argument("-o", "--output_name", required=False,
                        help="name of the output figure")
    parser.add_argument("--loss", action='store_true', default=False)
    args = parser.parse_args()


    loss_flag = args.loss
    print(f"Loss flag is {loss_flag}")
    if not loss_flag:
        figname = "Accuracy (%)"
        save_as = "acc.png"
    else:
        figname = "Loss"
        save_as = "loss.png"

    ids = []
    ids.append("run_0")
    ids.append("run_0")
    ids.append("run_0")

    names = []
    names.append(args.name_1)
    names.append(args.name_2)
    names.append(args.name_3)

    full_paths = []
    full_paths.append(os.path.join(path2exps, args.mdl_folder1))
    full_paths.append(os.path.join(path2exps, args.mdl_folder2))
    full_paths.append(os.path.join(path2exps, args.mdl_folder3))

    
    for i_cnt, (idx, path) in enumerate(zip(ids, full_paths)):
        for dirpath, dirnames, files in os.walk(path):
            for f in files:
                if idx in f:
                    full_paths[i_cnt] = os.path.join(full_paths[i_cnt], f)
                    break


    paths = []
    paths.append((full_paths[0], ids[0], '-', names[0]))
    paths.append((full_paths[1], ids[1], '-*', names[1]))
    paths.append((full_paths[2], ids[2], '--', names[2]))
    
    # paths = [(path1, "no-drop", "-") ,
    #          (path2, "drop", "--"),
    #          (path3, "drop", "-*")]
    
    skip_val = False

    # path1 = "experiments/CIFAR10/_plain_cifar_0.01/CNN2D__plain_cifar_0.01_run_0.json"
    # path2 = "experiments/CIFAR10/_dropout_0.5_0.01/CNN2D__dropout_0.5_0.01_run_0.json"
    # path3 = "experiments/CIFAR10/_smart_0.001_0.00004/CNN2D__smart_0.001_0.00004_run_0.json"
    
    # paths = [(path1, "no-drop", "-"),
    #          (path2, "drop", "--"),
    #          (path3, "smart-drop", "-.")]
    
    mpl.style.use("ggplot")

    fig, ax = plt.subplots()
    # fig_acc, ax_acc = plt.subplots()
    # fig_loss, ax_loss = plt.subplots()
    plt.yscale('linear')
    plt.xscale('linear')
    connectionstyle='angle, angleA=-120, angleB=180, rad=10'
    line = '-'
    thresh = 40
    for pth, nm, line, name in paths:
        with open (pth, "r") as fd:
            summary = json.load(fd)

        losses = summary["loss"]
        accuracies = summary["acc"]
        best_epoch = summary["best_epoch"]
        test_acc = summary["test_acc"]
        print(len(accuracies["train"]))
        #if len(accuracies['train']) > thresh:
        # else:
        #     t = np.arange(0, len(accuracies["train"]))
        # ax.annotate(str(test_acc),
        #             xy=(best_epoch, test_acc),
        #             xytext=(best_epoch - 4, 15 + test_acc),
        #             arrowprops=dict(arrowstyle="->",
        #                             connectionstyle=connectionstyle,
        #                             facecolor='white',
        #                             shrink=0.05))

        
        if loss_flag:
            flat_list = []
            for sub in losses["val"]:
                flat_list.extend(sub)
            losses["val"] = flat_list
            sampler_steps = 500
            t = np.arange(0, thresh * sampler_steps)
            total_val = len(losses["val"])
            total_train = len(losses["train"])
            keep = round(total_train/total_val)
            
            train_loss = []
            for k_ch in range(total_val):
                train_loss.append(losses["train"][k_ch * keep])
            
            # losses["train"] = train_loss
            
            import pdb; pdb.set_trace()
            
            plot_dict(ax, t, losses, name, line, skip_val, thresh=thresh * sampler_steps)
        else:
            t = np.arange(0, thresh)
            plot_dict(ax, t, accuracies, name, line, skip_val, thresh=thresh)
        
    plt.xlabel('Epochs')
    
    plt.ylabel(figname)
    plt.savefig(save_as, bbox_inches='tight')
    
    # fig_loss.ylabel(loss_figname)
    # fig_loss.savefig(loss_save_as, bbox_inches='tight')