# proof of concept

import pandas as pd
from sklearn import svm
from inference import Inference
import numpy as np


# given a positive array
# returns the values between 0 and 1
def normalize(df):
    return df / df.max()


def quantize(df):
    mid_th = 0.2
    high_th = 0.6
    x = df.values
    x[(x > high_th) & (x <= 1.0)] = 3
    x[(x > mid_th) & (x <= high_th)] = 2
    x[(x > 0.0) & (x <= mid_th)] = 1
    return pd.DataFrame(x)


# hyper-parameters
act_threshold = 0.0

# load input data
train_data_file = 'activations_data/DNN_cont_training_data.csv'
test_data_file = 'activations_data/DNN_cont_test_data.csv'

df_train = pd.read_csv(train_data_file, header=None)
df_test = pd.read_csv(test_data_file, header=None)

y_train = df_train.iloc[:, 110]
X_train = df_train.iloc[:, :110]

y_test = df_test.iloc[:, 110]
X_test = df_test.iloc[:, :110]

# applying ReLU
X_train = X_train.where(X_train > act_threshold, 0)
X_test = X_test.where(X_test > act_threshold, 0)

# convert remaining values to the range (0,1)
X_train = normalize(X_train)
X_test = normalize(X_test)

# quantize data to belong to {0,1,2,3}
X_train = quantize(X_train)
X_test = quantize(X_test)

print(X_train.tail())
print(X_test.tail())

SVM = svm.LinearSVC()
SVM.fit(X_train, y_train)

metaclf_pred = SVM.predict(X_test)
metaclf_errors = y_test.values - metaclf_pred
metaclf_errors[metaclf_errors != 0] = 1

train_score = SVM.score(X_train, y_train)
score = SVM.score(X_test, y_test)

print('Score of NACT (SVM) at the training set is: {:.4f}'.format(train_score))
print('Score of NACT (SVM) at the test set is: {:.4f}'.format(score))

initial_model_acc, initial_model_errors = Inference('DNN').get_performance_details()
initial_model_errors[initial_model_errors != 0] = 1

diff_errors = initial_model_errors.flatten() - metaclf_errors

count_errors_initial = np.count_nonzero(initial_model_errors == 1)
count_errors_nact = np.count_nonzero(metaclf_errors == 1)
count_diff_errors = np.count_nonzero(diff_errors != 0)
nact_corrections = np.count_nonzero(diff_errors == 1)

print('Total errors of initial DNN model: {}'.format(count_errors_initial))
print('Total errors of NACT model: {}'.format(count_errors_nact))
print('Number of different errors between models: {}'.format(count_diff_errors))
print('Number of NACT errors correctly classified by initial model: {}'.format(np.count_nonzero(diff_errors == -1)))
print('Number of initial model errors correctly classified by NACT: {} or {:.2f}%'.format(
    nact_corrections, 100. * nact_corrections / count_errors_initial))


# deprecated stuff

# from sklearn import preprocessing
# def normalize_by_column(df):
#     x = df.values  # returns a numpy array
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(x)
#     return pd.DataFrame(x_scaled)

# def normalize(df):
#     return (df - df.mean()) / (df.max() - df.min())

# hyper-parameters
# specified boundaries to trim the data values
# bounds = {
#     'lower': 0.15,
#     'upper': None,
# }
# act_threshold = 0.0

# X_test_a = normalize_by_column(X_test)
# X_train_a = normalize_by_column(X_train)

# trim values to the specified boundaries
# X_test = X_test.clip(lower=bounds['lower'], upper=bounds['upper'])
# X_train = X_train.clip(lower=bounds['lower'], upper=bounds['upper'])

# replace the lower values with zero
# X_train = X_train.where(X_train > bounds['lower'], 0)
# X_test = X_test.where(X_test > bounds['lower'], 0)

# Quantile-based discretization function, axis=0 per feature-neuron and axis=1 per pattern
# n_cuts = 2
# X_train = X_train.apply(lambda x: pd.qcut(x, n_cuts, labels=list(range(n_cuts))), axis=0)
# X_test = X_test.apply(lambda x: pd.qcut(x, n_cuts, labels=list(range(n_cuts))), axis=0)

# print(X_train.describe())
# print(X_test.describe())
