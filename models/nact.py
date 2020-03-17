import torch.optim as optim
from model_utils import get_dataframes_by_option
from config import Config
from torch.utils.data import DataLoader
import os
from dataloaders import DatasetNACT
from models import CNN2D
from model_utils import train, test, prepare_experiment_logs
from inference import Inference
from mnist import MNIST

if __name__ == '__main__':
    # get configuration settings
    config = Config()
    # override specific settings
    config.epochs = 30
    config.momentum = 0.9

    # number of activation features
    n_features_l = [110, 220, 440]

    for n_features in n_features_l:
        # override respective global setting
        config.n_acts = n_features

        if n_features == 110:
            input_shape = [10, 11]
        elif n_features == 220:
            input_shape = [20, 11]
        elif n_features == 440:
            input_shape = [20, 22]

        # experiment parameters
        options = ['binary', 'quantized', 'cont_normalized_threshold', 'continuous_raw']
        act_threshold = 0.2  # default value is 0
        quant_limits = [0.35, 0.5, 0.65]
        kernels = [40, 100]
        kernel_size = 2
        specific_identifier = 'CNN2D_' + str(n_features) + '_' + str(kernels[0]) + '-' + str(kernels[1])

        # specify data files
        train_data_file = config.ROOT_DIR + '/' + config.activations_data_dir + '/DNN_cont_training_data_' + str(n_features) + str(config.model_id) + '.csv'
        test_data_file = config.ROOT_DIR + '/' + config.activations_data_dir + '/DNN_cont_test_data_' + str(n_features) + str(config.model_id) + '.csv'

        # manage experiment logs
        current_filename = os.path.splitext(__file__)[0]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir, results_file = prepare_experiment_logs(current_dir, current_filename, specific_identifier)

        # performance of trained initial model
        saved_model_path = config.ROOT_DIR + '/saved_models/mnist_DNN_' + str(n_features) + str(config.model_id) + '.pt'
        mnist = MNIST()
        # get initial model performance
        infer = Inference(config.use_model)
        infer.get_performance_details()

        print('================== CNN2D on Activations data ===================')
        print('========================= Parameters =========================')
        print('act_threshold (except raw): {}, quant_limits: {}, kernels: {}, training_data: {}, kernel_size: {}'.
              format(act_threshold, ','.join(str(i) for i in [act_threshold] + quant_limits),
                     ','.join(str(i) for i in kernels), os.path.basename(train_data_file), kernel_size))

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
            model = CNN2D(input_shape, kernels, kernel_size).to(config.device)
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

            # training
            for epoch in range(1, config.epochs + 1):
                train(config, model, train_loader, optimizer, epoch)
                test(config, model, test_loader)

        print('================== Same CNN model on initial data (not activations data) ===================')
        # define model
        input_shape = [28, 28]  # each dimension's size
        model = CNN2D(input_shape, kernels, kernel_size).to(config.device)
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
        train_loader = mnist.get_train_loader()
        test_loader = mnist.get_test_loader()

        # training
        for epoch in range(1, config.epochs + 1):
            train(config, model, train_loader, optimizer, epoch)
            test(config, model, test_loader)
