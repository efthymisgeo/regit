import torch.optim as optim
from config import Config
import os
from models import CNN2D
from model_utils import train, test, validate, EarlyStopping, prepare_experiment_logs
from mnist import MNIST
import torch
import json


def run_training(model, config, train_loader, val_loader, test_loader,
                 optimizer_id="SGD", model_setup="bin", input_shape=[28,28],
                 kernels=[40, 100], kernel_size=5,
                 fc_layers = [1000, 2000, 2000, 1000, 800, 10],
                 bin_threshold=0.1, binarization_rate=0.1,
                 add_dropout=True,
                 p_drop=0.2, anorthosis=False):
    
    print("Model {} with {} optimizer".format(model_setup, optimizer_id))

    if optimizer_id is "SGD":
        # pick the optimizersd characteristics
        if model_setup is "bin":
            sgd_lr = config.lr_sgd_slow
            sgd_momentum = config.momentum
        elif model_setup is "raw":
            sgd_lr = config.lr
            sgd_momentum = config.momentum_raw
        else:
            raise ValueError("Not a valid model")
        # pick optimizer
        optimizer = optim.SGD(model.parameters(),
                              lr=sgd_lr,
                              momentum=sgd_momentum)
    elif optimizer_id is "Adam":
        print("Adam Optimizer")
        if model_setup is "bin":
            adam_lr = config.lr_adam
        elif model_setup is "raw":
            adam_lr = config.lr_adam_fast
        else:
            raise ValueError("Not a valid model")
        # pick optimizer
        optimizer = optim.Adam(model.parameters(),
                               lr=adam_lr)
    else:
        raise ValueError("Not a valid optimizer")

    # early stopping
    earlystop = EarlyStopping(patience=config.patience,
                              verbose=False,
                              config=config,
                              model_id=config.model_id)
    # training
    for epoch in range(1, config.epochs + 1):
        print("Epoch: [{}/{}]".format(epoch, config.epochs))
        train(config, model, train_loader, optimizer, epoch)
        val_loss, _ = validate(config, model, val_loader)
        earlystop(val_loss, model)
        if earlystop.early_stop:
            print("Early Stopping Training")
            break
        if epoch % 5 == 0:
            test(config, model, test_loader)
    print("finished training")
    print("Model Performance")
    saved_model = CNN2D(input_shape, kernels, kernel_size, fc_layers,
                        regularization=False, bin_threshold=bin_threshold,
                        binarization_rate=binarization_rate,
                        add_dropout=add_dropout,
                        p_drop=p_drop,
                        anorthosis=anorthosis).to(config.device)
        
    saved_model.load_state_dict(torch.load(config.saved_model_path,
                                           map_location=config.device))
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
    specific_identifier = '_' + nact_model + '_' + str(config.model_id) \
                          + '_' + str(kernels[0]) + '-' + str(kernels[1]) \
                          + '_' + '-'.join(str(i) for i in fc_layers)

    # manage experiment logs
    current_filename = os.path.splitext(__file__)[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir, results_file = prepare_experiment_logs(current_dir,
                                                        current_filename,
                                                        specific_identifier)

    print("================== CNN reverse end2end - regularization via binarization =="
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
        print("--------- Training showing bin data ---------")
        
        # define model
        model = CNN2D(input_shape, kernels, kernel_size, fc_layers,
                      regularization=False, bin_threshold=bin_threshold,
                      binarization_rate=binarization_rate,
                      add_dropout=add_dropout,
                      p_drop=p_drop,
                      anorthosis=use_anorthosis).to(config.device)

        if i == 0 and os.path.exists(config.saved_model_path):
            print('Continue training loading previous model')
            model.load_state_dict(torch.load(config.saved_model_path,
                                             map_location=config.device))
        elif i > 0:
            model.load_state_dict(torch.load(config.saved_model_path,
                                             map_location=config.device))
        
        # training
        bin_accuracy = run_training(model, config, train_loader, val_loader,
                                    test_loader, optimizer_id=config.optimizer,
                                    model_setup="bin", input_shape=input_shape,
                                    kernels=kernels, kernel_size=kernel_size,
                                    fc_layers=fc_layers,
                                    bin_threshold=bin_threshold,
                                    binarization_rate=binarization_rate,
                                    add_dropout=add_dropout,
                                    p_drop=p_drop, anorthosis=use_anorthosis)
            
        bin_acc_dict[str(i)] = bin_accuracy

        print("Maximum accuracy for regularized raw model is: {}"
              .format(max_bin_acc))
        
        import pdb; pdb.set_trace()
        
        if raw_mnist_only_flag is False:
            print("--------- Training showing raw data ---------")
            # define model
            model = CNN2D(input_shape, kernels, kernel_size, fc_layers,
                          regularization=False, bin_threshold=bin_threshold,
                          binarization_rate=binarization_rate,
                          add_dropout=add_dropout,
                          p_drop=p_drop,
                          anorthosis=not(use_anorthosis)).to(config.device)

            print("Loading Raw from Bin Trained Model")
            model.load_state_dict(torch.load(config.saved_model_path,
                                             map_location=config.device))
            # for debuginh purpose - remove later
            print("Regulariztion is {}".format(model.regularization))
        
            # training
            raw_accuracy = run_training(model, config, train_loader, 
                                        val_loader,
                                        test_loader,
                                        optimizer_id=config.optimizer,
                                        model_setup="raw",
                                        input_shape=input_shape,
                                        kernels=kernels,
                                        kernel_size=kernel_size,
                                        fc_layers=fc_layers,
                                        bin_threshold=bin_threshold,
                                        binarization_rate=binarization_rate,
                                        add_dropout=add_dropout,
                                        p_drop=p_drop,
                                        anorthosis=not(use_anorthosis))
        
            if raw_accuracy > max_raw_acc:
                max_raw_acc = raw_accuracy

            raw_acc_dict[str(i)] = raw_accuracy

            if raw_accuracy > max_raw_acc:
                max_raw_acc = raw_accuracy
                torch.save(model.state_dict(),
                           config.ROOT_DIR + '/saved_models/mnist_' \
                           + config.use_model + '_' \
                           + str(config.model_id) + '_best' + '.pt')
        

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
