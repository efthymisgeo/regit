import os
import sys
import copy
import json
import argparse
import collections
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))


def viz_single_distribuiton(ax, p_in):
    # matplotlib histogram
    ax.hist(p_in,
            color='blue',
            edgecolor ='black',
            bins=n_bins)

    # seaborn histogram
    sns.distplot(p_in,
                 hist=True,
                 kde=False, 
                 bins=n_bins,
                 color = 'blue',
                 hist_kws={'edgecolor':'black'})

def sort_dict_by_key(in_dict):
    # swap to int keys
    tmp_dict = {int(k):int(v) for k,v in in_dict.items()}
    od = collections.OrderedDict(sorted(tmp_dict.items()))
    return od

def viz_dict(ax, in_dict):
    ax.bar(in_dict.keys(),
           in_dict.values(),
           color='b')


if __name__ == "__main__":
    path = "experiments/CIFAR10/_smart_0.001_0.00004/CNN2D__smart_0.001_0.00004_run_0.json"
    with open(path, "r") as fd:
        summary = json.load(fd)
    switches = summary["switches"]
    
    fig, ax = plt.subplots()
    # Add labels
    for n_switch in switches:
        od_switch = sort_dict_by_key(n_switch)
        import pdb; pdb.set_trace()
        #viz_single_distribuiton(ax, n_switch)
        viz_dict(ax, od_switch)
        plt.title('Drop Distribution')
        plt.xlabel('Neuron')
        plt.ylabel('Switches')
        plt.savefig("dist.png", bbox_inches='tight')





