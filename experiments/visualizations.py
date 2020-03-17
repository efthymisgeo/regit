from config import Config
import os
from model_utils import visualize_activations
from mnist import MNIST


if __name__ == '__main__':
    config = Config()

    # experiment parameters - override settings
    n_features = 440
    config.layers = [215, 215, 10]
    config.n_acts = n_features
    act_threshold = 0.2  # default value is 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_filename = os.path.splitext(__file__)[0]
    results_dir = current_dir + '/results/' + current_filename
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_data_file = config.activations_data_dir + '/DNN_cont_training_data_' + str(n_features) + str(config.model_id) + '.csv'
    test_data_file = config.activations_data_dir + '/DNN_cont_test_data_' + str(n_features) + str(config.model_id) + '.csv'

    if not os.path.exists(train_data_file):
        config.acts_train_file = train_data_file
        config.acts_test_file = test_data_file
        MNIST(config).generate_activations_data()

    # visualize activations
    visualize_activations(config, results_dir, train_data_file, test_data_file, act_threshold)
