from model_utils import EuclideanClassifier, prepare_experiment_logs, get_dataframes_by_option
from config import Config
import os


if __name__ == '__main__':
    config = Config()

    n_acts_l = [110, 220, 440, 784, 1225]
    act_threshold = 0.2

    # manage experiment logs
    current_filename = os.path.splitext(__file__)[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    specific_identifier = '_' + '_'.join(str(i) for i in n_acts_l)
    results_dir, results_file = prepare_experiment_logs(current_dir, current_filename, specific_identifier)
    print('================== Euclidean classifier on representatives coming from binary data ==================\n')
    print('Activation threshold used to get binary vector for each sample: {}\n'.format(act_threshold))

    for n_acts_ in n_acts_l:
        config.n_acts = n_acts_
        train_data_file = config.activations_data_dir + '/DNN_cont_training_data_' + str(config.n_acts) + '.csv'
        test_data_file = config.activations_data_dir + '/DNN_cont_test_data_' + str(config.n_acts) + '.csv'

        X_train, X_test, y_train, y_test = get_dataframes_by_option('binary', train_data_file, test_data_file, config, act_threshold=act_threshold)

        # Classifier
        eucl = EuclideanClassifier(train_data_file, test_data_file, config.n_acts, act_threshold)
        eucl.fit()
        score = eucl.score(X_test, y_test)
        print('training data: {}, score at test set: {}'.format(os.path.basename(train_data_file), score))

        # without the data comes from the decision layer
        X_test_h = X_test.iloc[:, :n_acts_-10]
        # Classifier
        eucl_h = EuclideanClassifier(train_data_file, test_data_file, config.n_acts, act_threshold)
        eucl_h.fit(ignore_last_layer=True)
        score_h = eucl_h.score(X_test_h, y_test)
        print('using only hidden layers, score at test set: {}\n'.format(score_h))
