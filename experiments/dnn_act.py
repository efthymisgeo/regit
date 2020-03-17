import torch.optim as optim
from model_utils import get_dataframes_by_option
from config import Config
from torch.utils.data import DataLoader
import os
from dataloaders import DatasetNACT
from models import DNN
from model_utils import train, test, prepare_experiment_logs
from inference import Inference
from mnist import MNIST
import numpy as np


if __name__ == '__main__':
    # get configuration settings
    config = Config()

    # experiment parameters
    # valid options: 'binary', 'quantized', 'cont_normalized_threshold', 'continuous_raw'
    options = ['binary']
    layers = [1000, 2000, 2000, 1000, 800, 10]
    act_thresholds = np.arange(0.11, 0.30, 0.01).tolist()

    quant_limits = [0.35, 0.5, 0.65]
    specific_identifier = '_' + str(config.n_acts) + str(config.model_id) + '_straight_normalization_binary_various_thresholds_' + '-'.join(str(i) for i in layers)

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

    print('================== DNN on Activations data ===================')
    for act_threshold in act_thresholds:
        print('========================= Parameters =========================')
        print('act_threshold (except raw): {}, quant_limits: {}, layers: {}, training_data: {}'.
              format(act_threshold, ','.join(str(i) for i in [act_threshold] + quant_limits),
                     ','.join(str(i) for i in layers), os.path.basename(train_data_file)))

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
            model = DNN(config.n_acts, layers).to(config.device)
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

            # training
            for epoch in range(1, config.epochs + 1):
                train(config, model, train_loader, optimizer, epoch)
                test(config, model, test_loader)

    print('================== Same DNN model on initial data (not activations data) ===================')
    # define model
    mnist_input_size = 784
    model = DNN(mnist_input_size, layers).to(config.device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    train_loader = mnist.get_train_loader()
    test_loader = mnist.get_test_loader()

    # training
    for epoch in range(1, config.epochs + 1):
        train(config, model, train_loader, optimizer, epoch)
        test(config, model, test_loader)
