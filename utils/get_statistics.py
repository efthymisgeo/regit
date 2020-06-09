"""Script that takes the given accuracies per run and reports the corresponding
statistics mean, median and std. It also outputs in the same given folder a
violinplot which illustrates the nature of the experiment's performance
distribution over multiple runs
"""
import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))

if __name__ == "__main__":
    path2exps = "experiments/CIFAR10/"
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mdl_folder", required=True,
                        help="Path to model experiment file")
    args = parser.parse_args()

    result_folder = os.path.join(path2exps, args.mdl_folder)
    for dirpath, dirnames, files in os.walk(result_folder):
        for f in files:
            if ".csv" in f:
                csv_results = f
                break

    header = ["runs", "accuracy"]
    acc = []
    violi = []
    with open(os.path.join(result_folder, csv_results), "r") as fd:
        csv_reader = csv.reader(fd, delimiter=',')
        csv_reader.__next__()
        for row in csv_reader:
            violi.append(row)
            acc.append(float(row[1]))
    
    mean = np.mean(acc)
    std = np.std(acc)
    median = np.median(acc)
    q1 = np.percentile(acc, 25)
    q3 = np.percentile(acc, 75)
    iq = q3 - q1
    print(f"25th percentile is {q1}")
    print(f"75th percentile is {q3}")
    print(f"iq distance is is {iq}")
    print(f"inner fence from {q1 - 1.5*iq} to {q3 + 1.5*iq}")
    print(f"outter fence from {q1 - 3*iq} to {q3 + 3*iq}")

    h_mild_outliers = []
    l_mild_outliers = []
    h_extreme_outliers = []
    l_extreme_outliers = []
    for run_i in acc:
        if run_i <= (q1 - 3*iq):
            l_extreme_outliers.append(run_i)
            l_mild_outliers.append(run_i)
        elif run_i <= (q1 - 1.5*iq):
            l_mild_outliers.append(run_i)
        elif run_i >= (q3 + 3*iq):
            h_mild_outliers.append(run_i)
            h_extreme_outliers.append(run_i)
        elif run_i >= (q3 + 1.5*iq):
            h_mild_outliers.append(run_i)

    h_mild_p = 100.0 * len(h_mild_outliers) / len(acc)
    l_mild_p = 100.0 * len(l_mild_outliers) / len(acc)
    h_extreme_p = 100.0 * len(h_extreme_outliers) / len(acc)
    l_extreme_p = 100.0 * len(l_extreme_outliers) / len(acc)

    print(f"Mean is {np.mean(acc)} with std {np.std(acc)} \n"
          f"median {np.median(acc)} \n"
          f"High mild outliers {h_mild_p} \n"
          f"Low mild outliers {l_mild_p} \n"
          f"High extreme outliers {h_extreme_p} \n"
          f"Low extreme outliers {l_extreme_p} \n")
    
    plot_name = args.mdl_folder + "_violinplot.png"
    
    
    
    pd_acc = pd.DataFrame({'Accuracy': acc})
    
    ### Plotly script (issues while saving image)
    ### investigate plotly.offline
    # fig = px.violin(pd_acc,
    #                 y="Accuracy",
    #                 box=True, # draw box plot inside the violin
    #                 points='all', # can be 'outliers', or False
    #                 )
    # fig.write_image(plot_name)


    ### Matplotlib script 
    # fig, ax = plt.subplots()
    # ax.violinplot(acc,
    #               points=60,
    #               widths=0.7,
    #               showmeans=True,
    #               showextrema=True,
    #               showmedians=True,
    #               bw_method=0.5)
    # fig.savefig(plot_name, bbox_inches='tight')

    ### Seaborn script (does not work)
    sns.set(style="whitegrid")
    sns_plot = sns.violinplot(y="Accuracy",
                              data=pd_acc,
                              scale="width",
                              inner="point")
    #sns_plot.figure.savefig(plot_name) #, bbox_inches='tight')
    sns_plot.figure.savefig(os.path.join(result_folder, plot_name))
