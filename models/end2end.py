import torch.optim as optim
from config import Config
import os
from models import CNN2D
from model_utils import train, test, validate, EarlyStopping, \
    prepare_experiment_logs, LinearScheduler, MultiplicativeScheduler, \
    StepScheduler, ExponentialScheduler 
from mnist import MNIST
import torch
import json
#from torch.utils.tensorboard import SummaryWriter


def run_training(model, config, train_loader, val_loader, test_loader, attributor):
    """add docstrings
    """
    
    print(f"Model {config.use_model} with {config.optimizer} optimizer and"
          f"{config.lr} learning rate")
    
    if config.optimizer is "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=config.lr,
                              momentum=config.momentum)
    elif optimizer_id is "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        raise ValueError("Not a valid optimizer")

    if config.prob_scheduler is not None:
        epoch_steps = len(train_loader)  # n_steps the optimizer is being used 
        if config.prob_scheduler is "Lin":
            # this setup saturates at epoch 5
            saturation_epoch = 5
            p_schedule = \
                LinearScheduler([0.0, config.p_drop], saturation_epoch * epoch_steps)
        elif config.prob_scheduler is "Mul":
            saturation_epoch = 8
            p_schedule = \
                MultiplicativeScheduler([0.0, config.p_drop],
                                        saturation_epoch * epoch_steps)
        elif config.prob_scheduler is "Exp":
            print("##########################################################")
            print(f"Exponential Scheduler will be used with gamma {config.gamma}")
            print("##########################################################")
            saturation_epoch = 50
            p_schedule = \
                ExponentialScheduler([0.0, config.p_drop],
                                     saturation_epoch * epoch_steps,
                                     config.gamma)
        elif config.prob_scheduler is "Step":
            # TODO needs to be updated
            pass
        else:
            raise ValueError(f"{config.prob_scheduler} is not a valid scheduler."
                             "Refusing to proceed.")
    else:
        print("No custom scheduler is used. Proceeding without any.\n"
              "The dropout probability will be fixed from now on.")
        p_schedule = None

    # early stopping
    earlystop = EarlyStopping(patience=config.patience,
                              verbose=False,
                              config=config,
                              model_id=config.model_id)
    
    # add tensorboard functionality
    # create a summary writer using the specified folder name
    #writer = SummaryWriter("compare_experiments/" + "bindrop/" + config.model_id)
    writer=None
    
    # training
    for epoch in range(1, config.epochs + 1):
        print("Epoch: [{}/{}]".format(epoch, config.epochs))
        train_loss, train_acc = train(model,
                                      train_loader,
                                      optimizer,
                                      epoch,
                                      writer=writer,
                                      attributor=attributor,
                                      drop_scheduler=config.prob_scheduler,
                                      max_p_drop=config.p_drop,
                                      mix_rates=config.mixout,
                                      plain_drop_flag=config.plain_dropout_flag,
                                      p_schedule=p_schedule)
        
        #######################################################################
        ###### print frequencies of droping a neuron in the first FC layer
        #######################################################################
        from collections import OrderedDict
        from operator import itemgetter

        n_drops = OrderedDict(sorted(model.fc_1_idx.items(),
                              key=itemgetter(1), reverse=False))
        
        print(n_drops)

        #######################################################################

        #import pdb; pdb.set_trace()
        val_loss, val_acc = validate(config, model, val_loader,
                                     epoch=epoch, writer=writer)
        #writer.add_scalars("loss_curves", {"train": train_loss,
        #                                   "val": val_loss}, epoch-1)
        #writer.add_scalars("accuracy_curve", {"train": train_acc,
        #                                      "val": val_acc}, epoch-1)
        earlystop(val_loss, model)
        if earlystop.early_stop:
            print("Early Stopping Training")
            break
        if epoch % 5 == 0:
            test(config, model, test_loader)
    print("finished training")
    print("Model Performance")
    
    #writer.close()

    saved_model = CNN2D(input_shape=config.input_shape,
                        kernels=config.kernels,
                        kernel_size=config.kernel_size,
                        stride=config.stride,
                        padding=config.padding,
                        maxpool=config.maxpool,
                        pool_size=config.pool_size,
                        conv_drop=config.conv_drop,
                        conv_batch_norm=config.conv_batch_norm,
                        regularization=config.regularization,
                        activation=config.activation,
                        fc_layers=config.fc_layers,
                        add_dropout=config.add_dropout,
                        p_drop=config.p_drop,
                        device=config.device).to(config.device)
        
    saved_model.load_state_dict(torch.load(config.saved_model_path,
                                           map_location='cpu'))
    saved_model = saved_model.to(config.device)
    acc, _ = test(config, saved_model, test_loader)
    return acc


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
