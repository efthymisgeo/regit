from scipy.spatial import distance
from model_utils import get_binary_representatives, prepare_experiment_logs
from config import Config
import os
import numpy as np


if __name__ == '__main__':
    config = Config()

    n_features = 440
    act_threshold = 0.2

    # create results dir if doesn't exist
    current_filename = os.path.splitext(__file__)[0]

    # manage experiment logs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    specific_identifier = '_' + str(n_features)
    results_dir, results_file = prepare_experiment_logs(current_dir, current_filename, specific_identifier)

    train_data_file = config.activations_data_dir + '/DNN_cont_training_data_' + str(n_features) + '.csv'
    test_data_file = config.activations_data_dir + '/DNN_cont_test_data_' + str(n_features) + '.csv'

    nact_representative_b = get_binary_representatives(train_data_file, test_data_file, n_features, act_threshold)

    distances = np.zeros((10, 10))

    print('Jaccard distances of activations binary representatives')
    for i in range(10):
        for j in range(10):
            distances[i, j] = round(distance.jaccard(nact_representative_b[i], nact_representative_b[j]), 2)
        print(','.join(str(k) for k in distances[i, :]))
