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

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
    It must be "yes" (the default), "no" or None (meaning
    an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True,
             "y": True,
             "ye": True,
              "no": False,
              "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                            "(or 'y' or 'n').\n")

def csvDf(dat,**kwargs):
    from numpy import array
    data = array(dat)
    if data is None or len(data)==0 or len(data[0])==0:
        return None
    else:
        import pdb; pdb.set_trace()
        return pd.DataFrame(data[1:,1:],index=data[1:,0],columns=data[0,1:],**kwargs)

def makeDf(dat, index_dat, cols_dat):
    return pd.DataFrame(dat, index=index_dat, columns=cols_dat)


if __name__ == "__main__":
    path2exps = "experiments/CIFAR10/"
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--mdl_folder1", required=True,
                        help="Path to model experiment file")
    parser.add_argument("-m2", "--mdl_folder2", required=True,
                        help="Path to model experiment file")
    parser.add_argument("-m3", "--mdl_folder3", required=False,
                        help="Path to model experiment file")
    parser.add_argument("-m4", "--mdl_folder4", required=False,
                        help="Path to model experiment file")
    parser.add_argument("-o", "--output_name", required=True,
                        help="name of the output figure")
    args = parser.parse_args()
    
    mdl_list = []
    mdl_list.append(args.mdl_folder1)
    mdl_list.append(args.mdl_folder2)
    mdl_list.append(args.mdl_folder3)
    mdl_list.append(args.mdl_folder4)

    csv_list = []
    acc_list = []
    method_list = []    
    #names = ["Raw", "Dropout", "Condrop"]
    #names = ["peak-19", "peak-22", "peak-25", "peak-28"]
    #names = ["peak-19", "peak-22", "peak-25"]
    names = ["dropout", "buck_0.4-0.6", "buck_0.3-0.7", "buck_0.2-0.8"] #, "buck_0.1-0.9"] #, "sigma:0.001"]
    for i, mdl in enumerate(mdl_list):
        #import pdb; pdb.set_trace()
        result_folder = os.path.join(path2exps, mdl)
        for dirpath, dirnames, files in os.walk(result_folder):
            for f in files:
                if ".csv" in f:
                    csv_results = f
                    break
        csv_list.append(csv_results)

        acc = []
        #acc.append(names[i])
        with open(os.path.join(result_folder, csv_results), "r") as fd:
            csv_reader = csv.reader(fd, delimiter=',')
            csv_reader.__next__()
            for row in csv_reader:
                acc.append(float(row[1]))
                method_list.append(names[i])
        acc_list.extend(acc)
        #import pdb; pdb.set_trace()
        print(f"Mean is {np.mean(acc)} with std {np.std(acc)} and median {np.median(acc)}")
    
    
    plot_name = args.output_name + ".png"
    
    
    col_names = ["Accuracy", "Method"]
    d = {col_names[0]: acc_list,
         col_names[1]: method_list} 
    df = pd.DataFrame(data=d)
    
    sns.set(style="whitegrid")
    sns_plot = sns.violinplot(y=col_names[0],
                              x=col_names[1],
                              data=df,
                              scale="width",
                              inner="point")
    sns_plot.figure.savefig(plot_name) #, bbox_inches='tight')
    #sns_plot.figure.savefig("test.png")
