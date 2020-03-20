import os
import sys
import csv
import json
import torch
import numpy as np
import torch.optim as optim
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from modules.models import CNN2D
from utils.model_utils import train, test, validate, EarlyStopping
from utils.mnist import MNIST, CIFAR10
from utils.config_loader import print_config
from utils.opts import load_experiment_options
from models.end2end import run_training
from captum.attr import LayerConductance


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

    # architectural parameters
    # kernels = config.kernels
    # kernel_size = config.kernel_size
    # stride = config.stride
    # padding = config.padding
    # maxpool = config.maxpool
    # pool_size = config.pool_size
    # fc_layers = config.fc_layers
    # conv_drop = config.conv_drop
    # conv_batch_norm = config.conv_batch_norm
    # activation = config.activation
    # add_dropout = config.add_dropout
    # p_drop = config.p_drop
    # p_conv_drop = config.p_conv_drop
    input_shape = data_setup["input_shape"]
    runs = exp_setup["runs"]
    device = exp_setup["device"]

    print(f"ACTIVE DEVICE(S): {device}")
    # specific_identifier = '_' + nact_model + '_' + str(experiment_id) \
    #                       + '_' + str(kernels[0]) + '-' + str(kernels[1]) \
    #                       + '_' + '-'.join(str(i) for i in fc_layers)

    # manage experiment logs
    # current_filename = os.path.splitext(__file__)[0]
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # results_dir, results_file = prepare_experiment_logs(current_dir,
    #                                                     current_filename,
    #                                                     specific_identifier)

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
        gamma = exp_setup["use_drop_schedule"]["gamma"]
    print_config(exp_setup)

    # accumulate accuracies per run
    max_raw_acc = 0
    raw_acc_dict = []
    acc_list = []

    cnn_setup = model_setup["CNN2D"]
    fc_setup = model_setup["FC"]

    for i in range(runs):
        print("================== RUN {} ==================".format(i))
        print("--------- Training CNN-FC is about to take off ---------")
        
        # define model
        model = CNN2D(input_shape=input_shape,
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
                      device=device).to(device)

        if importance:
            attributor_list = \
                [LayerConductance(model, model.fc[i]) 
                    for i in range(len(fc_setup["fc_layers"])-1)]
        else:
            attributor_list = None
        
        for tag, value in model.named_parameters():
            print(tag)
            #if value.grad.size[1] != 0:
            #    print("grad_tag")
        print(model)
                         
        #if i == 0 and os.path.exists(config.saved_model_path):
        #    print('Continue training loading previous model')
        #    model.load_state_dict(torch.load(config.saved_model_path,
        #                                     map_location=config.device))
        #elif i > 0:
        #    model.load_state_dict(torch.load(config.saved_model_path,
        #                                     map_location=config.device))
        
        # training
        run_id = os.path.join(checkpoint_path, experiment_id + str(i)) 
        raw_accuracy = run_training(model,
                                    train_loader,
                                    val_loader,
                                    test_loader,
                                    exp_setup,
                                    model_setup,
                                    data_setup,
                                    attributor=attributor_list,
                                    ckpt_path=run_id)
        
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
    
    save_path = os.path.join(exp_setup["experiment_folder"],
                             data_setup["name"],
                             exp_setup["experiment_id"])
    
    try:
        os.mkdir(save_path)
    except OSError:
        print ("Creation of the directory %s failed" % save_path)
    else:
        print ("Successfully created the directory %s " % save_path)
    
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

    print("Statistical Accurary is ####### {} +/- {} ####### "
          "over {} runs".format(mean, std, runs))
    

        
