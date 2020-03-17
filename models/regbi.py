import os
import sys
import csv
import json
import torch
import numpy as np
import torch.optim as optim
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from configs import config
from models import CNN2D
from model_utils import train, test, validate, EarlyStopping, prepare_experiment_logs
from mnist import MNIST
from end2end import run_training
from captum.attr import LayerConductance


if __name__ == '__main__':
    nact_model = 'CNN2D'
    # get configuration settings
    config = config
    # When True the suggested regularization is not applied
    raw_mnist_only_flag = False
    model_id = config.model_id

    # architectural parameters
    kernels = config.kernels
    kernel_size = config.kernel_size
    stride = config.stride
    padding = config.padding
    maxpool = config.maxpool
    pool_size = config.pool_size
    fc_layers = config.fc_layers
    conv_drop = config.conv_drop
    conv_batch_norm = config.conv_batch_norm
    activation = config.activation
    add_dropout = config.add_dropout
    p_drop = config.p_drop
    p_conv_drop = config.p_conv_drop
    input_shape = config.input_shape
    runs = config.runs
    device = config.device

    print(f"ACTIVE DEVICE(S): {device}")
    specific_identifier = '_' + nact_model + '_' + str(model_id) \
                          + '_' + str(kernels[0]) + '-' + str(kernels[1]) \
                          + '_' + '-'.join(str(i) for i in fc_layers)

    # manage experiment logs
    current_filename = os.path.splitext(__file__)[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir, results_file = prepare_experiment_logs(current_dir,
                                                        current_filename,
                                                        specific_identifier)

    print("################## Regularization Experiment #####################")
    print("========================= Parameters =============================")
    print(f"~~~~~~~  p_drop {config.p_drop}      runs {config.runs} ~~~~~~~~~")

    # dataloaders
    mnist = MNIST(config)
    train_loader, val_loader = mnist.get_train_val_loaders()
    test_loader = mnist.get_test_loader()

    # experimental setup parameters
    regularization = config.regularization
    importance = config.importance  # True to use importance
    use_drop_schedule = config.use_drop_schedule  # True to use scheduler
    mixout = config.mixout
    plain_drop_flag = config.plain_dropout_flag
    custom_scheduler = config.prob_scheduler
    gamma = config.gamma
    

    # accumulate accuracies per run
    max_raw_acc = 0
    raw_acc_dict = []
    acc_list = []

    print(f"Regularization is {regularization} and Dropout is {config.add_dropout}")

    for i in range(runs):
        print("================== RUN {} ==================".format(i))
        print("--------- Training CNN-FC is about to take off ---------")
        
        # define model
        model = CNN2D(input_shape=input_shape,
                      kernels=kernels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      maxpool=maxpool,
                      pool_size=pool_size,
                      conv_drop=conv_drop,
                      p_conv_drop=p_conv_drop,
                      conv_batch_norm=conv_batch_norm,
                      regularization=regularization,
                      activation=activation,
                      fc_layers=fc_layers,
                      add_dropout=add_dropout,
                      p_drop=p_drop,
                      device=device).to(device)

        if importance:
            attributor_list = \
                [LayerConductance(model, model.fc[i]) 
                    for i in range(len(config.fc_layers)-1)]
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
        raw_accuracy = run_training(model,
                                    config,
                                    train_loader,
                                    val_loader,
                                    test_loader,
                                    attributor_list)
        
        acc_list.append(raw_accuracy)

        if raw_accuracy > max_raw_acc:
                max_raw_acc = raw_accuracy
                torch.save(model.state_dict(),
                           config.ROOT_DIR + '/saved_models/mnist_' \
                           + config.use_model + '_' \
                           + str(config.model_id) + '_best' + '.pt')
            
        raw_acc_dict.append({"run": str(i),
                             "accuracy": raw_accuracy}) 

        print("Maximum accuracy for regularized raw model is: {}"
              .format(max_raw_acc))
    
    save_path = os.path.join("experiments","results",config.model_id)
    
    try:
        os.mkdir(save_path)
    except OSError:
        print ("Creation of the directory %s failed" % save_path)
    else:
        print ("Successfully created the directory %s " % save_path)
    
    raw_name = "results_raw" + config.model_id + ".json"
    
    csv_columns = ["run", "accuracy"]
    csv_name = os.path.join(save_path, raw_name) 
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
    

        
