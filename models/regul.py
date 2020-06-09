import os
import sys
import csv
import json
import torch
import pandas as pd
import numpy as np
#import seaborn as sns
import torch.optim as optim
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

SEED_LIST = [1, 13, 77, 98, 66, 555, 327, 608, 235, 999, 2332, 2432, 10, 18, 12,
             89, 65, 43, 978, 998, 456, 876, 569, 976, 109]


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


if __name__ == '__main__':
    exp_config = load_experiment_options()
    model_setup = exp_config["model_opt"]
    exp_setup = exp_config["exp_opt"]
    data_setup = exp_config["data_opt"]

    model_name = exp_setup["model_name"]
    # this folder will contain all of the saved models for all runs of this experiment
    checkpoint_path = os.path.join(exp_setup["checkpoint_folder"],
                                   data_setup["name"],
                                   exp_setup["experiment_id"])
    _ = check_directory_and_create(checkpoint_path)
    experiment_id =  model_name + "_" + exp_setup["experiment_id"] + "_run_"

    input_shape = data_setup["input_shape"]
    runs = exp_setup["runs"]
    device = exp_setup["device"]

    print(f"ACTIVE DEVICE(S): {device}")
    print("################## Regularization Experiment #####################")
    print("========================= Parameters =============================")
    print_config(model_setup)

    # dataloaders
    print("======================= Dataset ==================================")
    if data_setup["name"] == "MNIST":
        data = MNIST(data_setup, exp_setup)
    elif data_setup["name"] == "CIFAR10":
        data = CIFAR10(data_setup, exp_setup)
    elif data_setup["name"] == "CIFAR100":
        #TODO
        pass
    else:
        raise NotImplementedError("Not a valid dataset")
    train_loader, val_loader = data.get_train_val_loaders()
    test_loader = data.get_test_loader()
    print_config(data_setup)

    # experimental setup parameters
    print("==================Experimental Parameters=========================")
    regularization = exp_setup["regularization"]
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
    print_config(exp_setup)

    # accumulate accuracies per run
    max_raw_acc = 0
    raw_acc_dict = []
    acc_list = []

    cnn_setup = model_setup["CNN2D"]
    fc_setup = model_setup["FC"]

    save_path = os.path.join(exp_setup["experiment_folder"],
                             data_setup["name"],
                             exp_setup["experiment_id"])

    # attribution config values
    attribute_setup = exp_setup.get("attribution", None)

    for i in range(runs):
        # dataloaders
        data_setup["seed"] = SEED_LIST[i]
        print("======================= Dataset ==================================")
        if data_setup["name"] == "MNIST":
            data = MNIST(data_setup, exp_setup)
        elif data_setup["name"] == "CIFAR10":
            data = CIFAR10(data_setup, exp_setup)
        elif data_setup["name"] == "CIFAR100":
            #TODO
            pass
        else:
            raise NotImplementedError("Not a valid dataset")
        train_loader, val_loader = data.get_train_val_loaders()
        test_loader = data.get_test_loader()
        print_config(data_setup)

        
        print("================== RUN {} ==================".format(i))
        print("--------- Training CNN-FC is about to take off ---------")
        
        # define model
        model = CNNFC(input_shape=input_shape,
                      kernels=cnn_setup["kernels"],
                      kernel_size=cnn_setup["kernel_size"],
                      stride=cnn_setup["stride"],
                      padding=cnn_setup["padding"],
                      maxpool=cnn_setup["maxpool"],
                      pool_size=cnn_setup["pool_size"],
                      conv_drop=cnn_setup["conv_drop"],
                      p_conv_drop=cnn_setup["p_conv_drop"],
                      conv_batch_norm=cnn_setup["conv_batch_norm"],
                      regularization=regularization,
                      activation=fc_setup["activation"],
                      fc_layers=fc_setup["fc_layers"],
                      add_dropout=fc_setup["fc_drop"],
                      p_drop=fc_setup["p_drop"],
                      idrop_method=map_rank_method,
                      p_buckets=p_buckets,
                      inv_trick=inv_trick,
                      pytorch_dropout=exp_setup["plain_drop"],
                      device=device).to(device)

        if importance:
            layer_list = []
            for i_fc in range(len(fc_setup["fc_layers"])-1):
                layer_list.append(model.fc[i_fc])
            attributor = Attributor(model, layer_list)
        else:
            attributor = None
        
        for tag, value in model.named_parameters():
            print(tag)
        print(model)
                         
        # training
        run_id = os.path.join(checkpoint_path, experiment_id + str(i)) 
        train_summary = run_training(model,
                                     train_loader,
                                     val_loader,
                                     test_loader,
                                     exp_setup,
                                     model_setup,
                                     data_setup,
                                     attribute_setup,
                                     attributor=attributor,
                                     ckpt_path=run_id)
        
        raw_accuracy = train_summary["test_acc"]
        acc_list.append(raw_accuracy)

        if raw_accuracy > max_raw_acc:
                max_raw_acc = raw_accuracy
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_path,
                                        experiment_id) \
                                        + '_best' + '.pt')
            
        raw_acc_dict.append({"run": str(i),
                             "accuracy": raw_accuracy}) 

        print("Maximum accuracy for regularized raw model is: {}"
              .format(max_raw_acc))
        
        try:
            os.mkdir(save_path)
        except OSError:
            print ("Creation of the directory %s failed" % save_path)
        else:
            print ("Successfully created the directory %s " % save_path)
            
        summary_json_name = os.path.join(save_path,
                                         experiment_id + str(i) + ".json")
        
        with open (summary_json_name, "w") as fd:
            json.dump(train_summary, fd)

    
    
    acc_csv_name = experiment_id + "all_accuracy.csv"
    
    csv_columns = ["run", "accuracy"]
    csv_name = os.path.join(save_path, acc_csv_name) 
    with open(csv_name, "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for data in raw_acc_dict:
            writer.writerow(data)
    
    print("Results succesfully save at {}".format(csv_name))

    mean = np.mean(acc_list)
    std = np.std(acc_list)
    median = np.median(acc_list)
    max_acc = max(acc_list)
    min_acc = min(acc_list)

    print(f"Statistical Accurary is ####### {mean} +/- {std} ####### "
          f"over {runs} runs, while median accuracy is {median} "
          f"Maximum {max_acc} and minimum {min_acc}")

    # plot_name = experiment_id + "_violinplot.png"
    # pd_acc = pd.DataFrame({'Accuracy': acc_list})
    # sns.set(style="whitegrid")
    # sns_plot = sns.violinplot(y="Accuracy",
    #                           data=pd_acc,
    #                           scale="width",
    #                           inner="point")
    # sns_plot.figure.savefig(os.path.join(save_path, plot_name))

    

        
