from model_utils import EuclideanClassifier, prepare_experiment_logs, get_dataframes_by_option
from config import Config
import os
import operator
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    config = Config()

    config.n_acts = 800
    act_threshold = 0.25
    model_ids = ['_fc_in', '_fc_in_2', '_fc_in_3']

    # manage experiment logs
    current_filename = os.path.splitext(__file__)[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    specific_identifier = '_' + str(config.n_acts)
    results_dir, results_file = prepare_experiment_logs(current_dir, current_filename, specific_identifier)
    print('================== Ensemble classifier using Euclideans trained on data from different initial model (CNN) runs ==================\n')
    print('Activation threshold used to get binary vector for each sample: {}\n'.format(act_threshold))
    print('Activations taken from models : {}\n'.format(','.join('CNN2D' + str(i) for i in model_ids)))

    eucl_classifiers = {}
    X_test = {}
    scores = {}
    for idx, model_id in enumerate(model_ids):
        train_data_file = config.activations_data_dir + '/CNN2D_cont_training_data_' + str(config.n_acts) + str(model_id) + '.csv'
        test_data_file = config.activations_data_dir + '/CNN2D_cont_test_data_' + str(config.n_acts) + str(model_id) + '.csv'

        _, X_test[idx], _, y_test = get_dataframes_by_option('binary', train_data_file, test_data_file, config, act_threshold=act_threshold)

        # Classifier
        eucl_classifiers[idx] = EuclideanClassifier(train_data_file, test_data_file, config.n_acts, act_threshold)
        eucl_classifiers[idx].fit()
        scores[idx] = eucl_classifiers[idx].score(X_test[idx], y_test)
        print('training data: {}, score at test set: {}'.format('CNN2D' + model_id, scores[idx]))

    # Ensemble - Caution: different generated activations data for each run/classifier
    predictions = {}
    best_clf_idx = max(scores.items(), key=operator.itemgetter(1))[0]
    for idx, eucl_classifier in enumerate(eucl_classifiers):
        predictions[idx] = eucl_classifiers[idx].predict(X_test[idx])

    # Hard Voting implementation
    ensemble_preds = []
    for i in range(len(X_test[0])):
        clfs_preds = [predictions[j][i] for j in range(len(model_ids))]
        if clfs_preds[0] == clfs_preds[1] or clfs_preds[0] == clfs_preds[2]:
            ensemble_preds.append(clfs_preds[0])
        elif clfs_preds[1] == clfs_preds[2]:
            ensemble_preds.append(clfs_preds[1])
        else:
            ensemble_preds.append(clfs_preds[best_clf_idx])

    print('Ensemble classifier score at test set (hard voting): {}'.format(accuracy_score(y_test, ensemble_preds)))
