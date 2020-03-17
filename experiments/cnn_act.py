import torch.optim as optim
from model_utils import get_dataframes_by_option
from config import Config
from torch.utils.data import DataLoader
import os
from dataloaders import DatasetNACT
from models import CNN1D, CNN2D
from model_utils import train, test, prepare_experiment_logs
from inference import Inference
from mnist import MNIST

if __name__ == '__main__':
    # get configuration settings
    config = Config()

    # experiment parameters
    options = ['binary', 'quantized', 'cont_normalized_threshold', 'continuous_raw']
    act_threshold = 0.2  # default value is 0
    quant_limits = [0.35, 0.5, 0.65]
    kernels = [40, 100]
    kernel_size = 5
    fc_layers = [1000, 2000, 2000, 1000, 800, 10]
    nact_model = 'CNN2D'
    specific_identifier = '_' + nact_model + '_' + str(config.n_acts) + str(config.model_id) + '_' + \
                          str(act_threshold).replace('.', '_') + '_' + str(kernels[0]) +\
                          '-' + str(kernels[1]) + '_' + '-'.join(str(i) for i in fc_layers)

    # create results dir if doesn't exist
    current_filename = os.path.splitext(__file__)[0]

    # manage experiment logs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir, results_file = prepare_experiment_logs(current_dir, current_filename, specific_identifier)

    # run training and activations data generation if not exist
    saved_model_path = config.ROOT_DIR + '/saved_models/mnist_' + config.use_model + '_' + str(config.n_acts) + str(config.model_id) + '.pt'
    mnist = MNIST(config)
    if not os.path.exists(saved_model_path):
        mnist.run_training()

    if not os.path.exists(config.acts_train_file):
        mnist.generate_activations_data()

    # get initial model performance
    infer = Inference(config.use_model)
    infer.get_performance_details()

    # override specific settings
    config.epochs = 50
    config.momentum = 0.9

    # specify data files
    train_data_file = config.acts_train_file
    test_data_file = config.acts_test_file

    print('================== CNN on Activations data ===================')
    print('========================= Parameters =========================')
    print('act_threshold (except raw): {}, quant_limits: {}, kernels: {}, training_data: {}, kernel_size: {}, fc_layers: {}'.
          format(act_threshold, ','.join(str(i) for i in [act_threshold] + quant_limits),
                 ','.join(str(i) for i in kernels), os.path.basename(train_data_file), kernel_size, ','.join(str(i) for i in fc_layers)))

    for option in options:
        print('\n==========================================================================================')
        print('----------------> data: ' + str(option))
        if option == 'continuous_raw':
            act_threshold = None
        # get dataframes
        X_train, X_test, y_train, y_test = get_dataframes_by_option(option, train_data_file, test_data_file, config,
                                                                    act_threshold=act_threshold, quant_limits=quant_limits)

        # load train-test dataset
        train_dataset = DatasetNACT(X_train, y_train)
        test_dataset = DatasetNACT(X_test, y_test)

        # dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True)

        # define model
        if nact_model == 'CNN2D':
            model = CNN2D(config.cnn2d_nact_input_shape, kernels, kernel_size, fc_layers).to(config.device)
        else:
            model = CNN1D(config.n_acts, kernels, kernel_size).to(config.device)
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

        # training
        for epoch in range(1, config.epochs + 1):
            train(config, model, train_loader, optimizer, epoch)
            test(config, model, test_loader)

    print('================== Same CNN model on initial data (not activations data) ===================')
    # define model
    input_shape = [28, 28]  # input dimensions
    model = CNN2D(input_shape, kernels, kernel_size, fc_layers).to(config.device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    train_loader = mnist.get_train_loader()
    test_loader = mnist.get_test_loader()

    # training
    for epoch in range(1, config.epochs + 1):
        train(config, model, train_loader, optimizer, epoch)
        test(config, model, test_loader)
