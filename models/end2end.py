import os 
import sys
import json
import torch
import torch.optim as optim
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utils.model_utils import train, test, validate, EarlyStopping, \
    LinearScheduler, MultiplicativeScheduler, \
    StepScheduler, ExponentialScheduler 
from utils.mnist import MNIST
from configs.config import Config
from modules.models import CNN2D

#from torch.utils.tensorboard import SummaryWriter


def run_training(model,
                 train_loader,
                 val_loader,
                 test_loader,
                 experiment_setup,
                 model_setup,
                 data_setup,
                 attributor=[],
                 ckpt_path="checkpoints/MNIST"):
    """ # TODO add docstrings
    """
    regularization = experiment_setup["regularization"]
    optim_setup = experiment_setup["optimization"]
    print(f"Model {experiment_setup['model_name']} with "
          f"{optim_setup['optimizer']} optimizer and "
          f"{optim_setup['lr']} learning rate")
    
    if optim_setup["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=optim_setup["lr"],
                              momentum=optim_setup["momentum"])
    elif optim_setup["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=optim_setup["lr"])
    else:
        raise NotImplementedError("Not a valid optimizer")

    use_optim_scheduler = optim_setup["scheduling"]
    if use_optim_scheduler:
        print("Adding ReduceLROnPlateu Scheduler")
        lr_scheduler = \
            optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'min',
                                                  factor=optim_setup["factor"],
                                                  patience=optim_setup["wait"],
                                                  verbose=True)

    drop_schedule_setup = experiment_setup["use_drop_schedule"]
    if drop_schedule_setup != {}:
        p_drop = drop_schedule_setup["p_drop"]
        epoch_steps = len(train_loader)  # n_steps the optimizer is being used 
        if drop_schedule_setup["prob_scheduler"] is "Lin":
            # this setup saturates at epoch 5
            saturation_epoch = drop_schedule_setup["saturation_epoch"]
            p_schedule = \
                LinearScheduler([0.0, p_drop],
                                saturation_epoch * epoch_steps)
        elif drop_schedule_setup["prob_scheduler"] == "Mul":
            saturation_epoch = 8
            p_schedule = \
                MultiplicativeScheduler([0.0, p_drop],
                                        saturation_epoch * epoch_steps)
        elif drop_schedule_setup["prob_scheduler"] == "Exp":
            print("##########################################################")
            print("Exponential Scheduler will be used with gamma "
                  f"{drop_schedule_setup['gamma']}")
            print("##########################################################")
            saturation_epoch = 100
            p_schedule = \
                ExponentialScheduler([0.0, p_drop],
                                     saturation_epoch * epoch_steps,
                                     drop_schedule_setup["gamma"])
        elif drop_schedule_setup["prob_scheduler"] == "Step":
            # TODO needs to be updated
            pass
        else:
            raise NotImplementedError(f"{drop_schedule_setup['prob_scheduler']}"
                                      "is not a valid scheduler. " 
                                      "Refusing to proceed.")
        use_inv_drop = drop_schedule_setup["use_inv_drop"]
        inv_startegy = drop_schedule_setup["inv_strategy"]
        reset_counter = not(drop_schedule_setup["track_history"])
    else:
        print("No custom scheduler is used. Proceeding without any.\n"
              "The dropout probability will be fixed from now on.")
        p_schedule = None
        p_drop = experiment_setup["p_drop"]
        use_inv_drop = False
        inv_startegy = None
        reset_counter = True


    # early stopping
    earlystop = EarlyStopping(patience=experiment_setup["patience"],
                              verbose=False,
                              save_model=experiment_setup["save_model"],
                              ckpt_path=ckpt_path) # needs to be model and run specific
    
    # add tensorboard functionality
    # create a summary writer using the specified folder name
    #writer = SummaryWriter("compare_experiments/" + "bindrop/" + config.model_id)
    writer=None

    test_freq = 10 # 3 for MNIST 5 for CIFAR
    loss_dict = {"train": [], "val": [], "test": []}
    acc_dict = {"train": [], "val": [], "test": []}
    gold_epoch_id = -1  # best epoch id
    p_drop_list = []  #  p_drop along epochs
    lr_list = []  # lr along epochs
    switches = []  # number of switches in a given layer

    if use_inv_drop:
        print("use inverted dropout strategy {inv_startegy}")
        if reset_counter:
            print("TEMPORAL HISTORY (PER EPOCH)")
        else:
            print("FULL HISTORY")

    if experiment_setup["importance"]:
        sampling_imp = experiment_setup["imp_sampling"]
    else:
        # sample at every batch. equivalent to no smapling
        sampling_imp = 1
    print(f"Sampling Neuron Importance per {sampling_imp} batces")

    aggregate = experiment_setup["aggregate"]
    if aggregate:
        print(f"Using a single mask per batch")
    else:
        print(f"Using a mask per sample in every batch")
        
    # training
    for epoch in range(1, experiment_setup["epochs"] + 1):
        print("Epoch: [{}/{}]".format(epoch, experiment_setup["epochs"]))
        train_loss, train_acc, p_list, train_loss_list = \
            train(model,
                  train_loader,
                  optimizer,
                  epoch,
                  regularization=regularization,
                  writer=writer,
                  attributor=attributor,
                  max_p_drop=p_drop,
                  mix_rates=experiment_setup["mixout"],
                  plain_drop_flag=experiment_setup["plain_drop"],
                  p_schedule=p_schedule,
                  use_inverted_strategy=use_inv_drop,
                  inverted_strategy=inv_startegy,
                  reset_counter=reset_counter,
                  sampling_imp=sampling_imp,
                  aggregate=aggregate)
        
        loss_dict["train"].extend(train_loss_list)
        acc_dict["train"].append(train_acc)
        p_drop_list.append(p_list)

        #######################################################################
        ###### print frequencies of droping a neuron in the first FC layer
        #######################################################################
        #from collections import OrderedDict
        #from operator import itemgetter

        #n_drops = OrderedDict(sorted(model.fc_1_idx.items(),
        #                      key=itemgetter(1), reverse=False))
        switches.append(model.switch_counter)
        
        #######################################################################

        #import pdb; pdb.set_trace()
        val_loss, val_acc, val_loss_list = validate(model,
                                                    val_loader,
                                                    epoch=epoch,
                                                    writer=writer)

        loss_dict["val"].append(val_loss_list)
        acc_dict["val"].append(val_acc)

        if use_optim_scheduler:
            lr_scheduler.step(val_loss)

        #writer.add_scalars("loss_curves", {"train": train_loss,
        #                                   "val": val_loss}, epoch-1)
        #writer.add_scalars("accuracy_curve", {"train": train_acc,
        #                                      "val": val_acc}, epoch-1)
        if epoch % test_freq == 0 or epoch == 1:
            test_loss, test_acc, _, test_loss_list =\
                test(model, test_loader)
            loss_dict["test"].extend(test_loss_list)
            acc_dict["test"].append(test_acc)
        
        earlystop(val_loss, model)
        if earlystop.early_stop:
            gold_epoch_id = earlystop.best_epoch_id
            print("Early Stopping Training")
            break
    
    print("Finished training")
    print("Model Performance")
    
    #writer.close()

    cnn_setup = model_setup["CNN2D"]
    fc_setup = model_setup["FC"]

    saved_model = CNN2D(input_shape=data_setup["input_shape"],
                        kernels=cnn_setup["kernels"],
                        kernel_size=cnn_setup["kernel_size"],
                        stride=cnn_setup["stride"],
                        padding=cnn_setup["padding"],
                        maxpool=cnn_setup["maxpool"],
                        pool_size=cnn_setup["pool_size"],
                        conv_drop=cnn_setup["conv_drop"],
                        p_conv_drop=cnn_setup["p_conv_drop"],
                        conv_batch_norm=cnn_setup["conv_batch_norm"],
                        regularization=experiment_setup["regularization"],
                        activation=fc_setup["activation"],
                        fc_layers=fc_setup["fc_layers"],
                        add_dropout=fc_setup["fc_drop"],
                        p_drop=fc_setup["p_drop"],
                        device=model.device).to(model.device)
        
    saved_model.load_state_dict(torch.load(ckpt_path + ".pt", map_location='cpu'))
    saved_model = saved_model.to(model.device)
    test_loss, test_acc, _, _ = test(saved_model, test_loader)
    
    train_summary = {"loss": loss_dict,
                     "acc": acc_dict,
                     "best_epoch": gold_epoch_id,
                     "p_drop": p_drop_list,
                     "switches": switches,
                     "test_acc": test_acc}

    return train_summary


if __name__ == '__main__':
    # get configuration settings
    config = Config()

    # CNN on MNIST without suggested regularization
    raw_mnist_only_flag = False

    # experiment parameters
    kernels = [40, 100]
    kernel_size = 5
    fc_layers = [1000, 2000, 2000, 1000, 800, 10]
    nact_model = 'CNN2D'
    bin_threshold = config.bin_threshold
    binarization_rate = config.binarization_rate
    add_dropout = config.add_dropout
    p_drop = config.p_drop
    use_anorthosis = config.anorthosis

    print(config.device)
    print(config.saved_model_path)

    specific_identifier = '_' + nact_model + '_' + str(config.model_id) \
                          + '_' + str(kernels[0]) + '-' + str(kernels[1]) \
                          + '_' + '-'.join(str(i) for i in fc_layers)

    # manage experiment logs
    current_filename = os.path.splitext(__file__)[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir, results_file = prepare_experiment_logs(current_dir,
                                                        current_filename,
                                                        specific_identifier)

    print("================== CNN end2end - regularization via binarization =="
          "=================")
    print('\n========================= Parameters =========================')
    print("bin_threshold: {}, binarization_rate: {}, kernels: {}" \
          "kernel_size: {}, fc_layers: {}".
          format(bin_threshold, binarization_rate, 
                 ','.join(str(i) for i in kernels),
                 kernel_size, ','.join(str(i) for i in fc_layers)))

    # dataloaders
    mnist = MNIST(config)
    train_loader, val_loader = mnist.get_train_val_loaders()
    test_loader = mnist.get_test_loader()
    
    # set number of training cycles
    cycles = 1

    # accumulate accuracies per cycle
    max_raw_acc = 0
    max_bin_acc = 0
    raw_acc_dict = {}
    bin_acc_dict = {}

    input_shape = [28, 28]  # input dimensions
    for i in range(cycles):
        print("================== Cycle {} ==================".format(i))
        print("--------- Training showing raw data ---------")
        
        # define model
        model = CNN2D(input_shape, kernels, kernel_size, fc_layers,
                      regularization=False, bin_threshold=bin_threshold,
                      binarization_rate=binarization_rate,
                      add_dropout=add_dropout,
                      p_drop=p_drop,
                      anorthosis=not(use_anorthosis)).to(config.device)
       
        if i == 0 and os.path.exists(config.saved_model_path):
            print('Continue training loading previous model')
            model.load_state_dict(torch.load(config.saved_model_path,
                                             map_location=config.device))
        elif i > 0:
            model.load_state_dict(torch.load(config.saved_model_path,
                                             map_location=config.device))
        
        # training
        raw_accuracy = run_training(model, config, train_loader, val_loader,
                                    test_loader, optimizer_id=config.optimizer,
                                    model_setup="raw", input_shape=input_shape,
                                    kernels=kernels, kernel_size=kernel_size,
                                    fc_layers=fc_layers,
                                    bin_threshold=bin_threshold,
                                    binarization_rate=binarization_rate,
                                    add_dropout=add_dropout,
                                    p_drop=p_drop,
                                    anorthosis=not(use_anorthosis))
        
        if raw_accuracy > max_raw_acc:
                max_raw_acc = raw_accuracy
                torch.save(model.state_dict(),
                           config.ROOT_DIR + '/saved_models/mnist_' \
                           + config.use_model + '_' \
                           + str(config.model_id) + '_best' + '.pt')
            
        raw_acc_dict[str(i)] = raw_accuracy

        if raw_mnist_only_flag is False:
            print("--------- Training showing binarized data ---------")
            # define model
            model = CNN2D(input_shape, kernels, kernel_size, fc_layers,
                          regularization=False, bin_threshold=bin_threshold,
                          binarization_rate=binarization_rate,
                          add_dropout=add_dropout,
                          p_drop=p_drop,
                          anorthosis=use_anorthosis).to(config.device)
            print("Loading Binary from Raw Trained Model")
            model.load_state_dict(torch.load(config.saved_model_path,
                                             map_location=config.device))
            # for debuginh purpose - remove later
            print("Regulariztion is {}".format(model.regularization))
        
            # training
            bin_accuracy = run_training(model, config, train_loader,
                                        val_loader, test_loader,
                                        optimizer_id=config.optimizer_bin,
                                        model_setup="bin",
                                        input_shape=input_shape,
                                        kernels=kernels,
                                        kernel_size=kernel_size,
                                        fc_layers=fc_layers,
                                        bin_threshold=bin_threshold,
                                        binarization_rate=binarization_rate,
                                        add_dropout=add_dropout,
                                        p_drop=p_drop,
                                        anorthosis=use_anorthosis)
        
            if bin_accuracy > max_bin_acc:
                max_bin_acc = bin_accuracy

            bin_acc_dict[str(i)] = bin_accuracy


        print("Maximum accuracy for raw model is: {} and for bin model is: {}"
              .format(max_raw_acc, max_bin_acc))
    
    save_path = os.path.join("experiments","results",config.model_id)
    
    try:
        os.mkdir(save_path)
    except OSError:
        print ("Creation of the directory %s failed" % save_path)
    else:
        print ("Successfully created the directory %s " % save_path)
    
    raw_name = "results_raw" + config.model_id + ".json"
    bin_name = "results_bin" + config.model_id + ".json"
    
    with open(os.path.join(save_path, raw_name), "w") as f:
        json.dump(raw_acc_dict, f)
    with open(os.path.join(save_path, bin_name), "w") as f:
        json.dump(bin_acc_dict, f)
