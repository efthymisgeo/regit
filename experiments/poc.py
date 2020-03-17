import pandas as pd
from sklearn import svm
from inference import Inference
from config import Config
from logger import Logger
import numpy as np
from matplotlib import pyplot as plt
from model_utils import convert_data_to_binary, convert_to_binary
import os
import sys


# Helper functions

def svm_classify(X_train, y_train, X_test, y_test):
    SVM = svm.LinearSVC()
    SVM.fit(X_train, y_train)

    train_score = SVM.score(X_train, y_train)
    score = SVM.score(X_test, y_test)
    
    return train_score, score


def plot_activations_on_window(start, stop, mean_nact_data, df_reprs, save_destination):
    colors = ['r', 'b', 'g', 'c', 'm', 'y', '#545812', '#737452', '#458612', '#191342', 'k']
    loc_step = 0.06
    
    plt.figure(figsize=(18,10))
    x = np.arange(start,stop,1)
    plt.stem(x+loc_step*10, mean_nact_data[start:stop], colors[-1], label='mean')
    
    for digit in range(10):
        data = df_reprs[df_reprs[110] == digit].iloc[:, start:stop].values.flatten()
        plt.stem(x+loc_step*digit, data, colors[digit], markerfmt='go', label=digit)

    plt.savefig(save_destination)


# create results dir if doesn't exist
current_filename = os.path.splitext(__file__)[0]
config = Config()
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = current_dir + '/results/' + current_filename
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# redirect to results file
sys.stdout = Logger(results_dir + '/results.txt')

# Experiment start
initial_model_acc, diff_pred_real = Inference('DNN').get_performance_details()


# load input data to dataframes
train_data_file = config.acts_train_file
test_data_file = config.acts_test_file

# apply a larger ReLU threshold: 0.2
act_threshold = 0.2

# convert data to binary vestors
X_train_b_act, X_test_b_act, y_train, y_test = convert_data_to_binary(train_data_file, test_data_file, act_threshold)


# SVM classifier using the above data
train_score, score = svm_classify(X_train_b_act, y_train, X_test_b_act, y_test)
print('Score of SVM at binary Activations data with ReLU threshold 0.2 at the training set is: {:.2f}%'
      ', and at the test set is: {:.2f}%'.format(train_score * 100, score * 100))

###############################################
# Visualizations of Activated nodes per class #
###############################################

# combine data to dataframe
nact_data = pd.concat([X_train_b_act, y_train], axis=1)

# create a dict to store the mean vector of each class
nact_representative = {}
for digit in range(10):
    nact_representative[digit] = nact_data[nact_data[110] == digit].iloc[:, :110].mean()


# convert each representative to a binary vector
nact_representative_b = {}

for digit in range(10):
    nact_representative_b[digit] = convert_to_binary(nact_representative[digit], 0.5)


# construct arrays for the visualizations
visualization = {}

for digit in range(10):
    vis_array = np.zeros((70, 96))
    vis_array[10:60, 25] = nact_representative_b[digit].iloc[0:50].values.flatten()
    vis_array[10:60, 50] = nact_representative_b[digit].iloc[50:100].values.flatten()
    vis_array[30:40, 75] = nact_representative_b[digit].iloc[100:110].values.flatten()
    visualization[digit] = vis_array

fig = plt.figure(figsize=(18,18))
ax = fig.subplots(nrows=5, ncols=2)

digit = 0
for row in ax:
    for col in row:
        col.imshow(visualization[digit])
        col.set_title('Digit ' + str(digit))
        digit += 1

plt.savefig(results_dir + '/activations_per_digit.png')


# combine the above to a single visualization array
visualization_c = np.zeros((70, 80))

step = 0
for digit in range(10):
    visualization_c[10:60, 5 + digit + step] = nact_representative_b[digit].iloc[0:50].values.flatten() * (digit + 1)
    visualization_c[10:60, 30 + digit + step] = nact_representative_b[digit].iloc[50:100].values.flatten() * (digit + 1)
    visualization_c[30:40, 55 + digit + step] = nact_representative_b[digit].iloc[100:110].values.flatten() * (digit + 1)
    step += 1

plt.figure(figsize=(18,18))
plt.imshow(visualization_c, cmap='gist_stern')
plt.gca().set_title('Digits Activations')

colorbar = plt.colorbar(boundaries=np.arange(0.5,11.5,1), orientation='horizontal', fraction=0.082, pad=0.0)

# colorbar labels
labels = np.arange(0,10,1)
loc = labels + 1
colorbar.set_ticks(loc)

colorbar.set_ticklabels(labels)

plt.savefig(results_dir + '/digits_activations.png')


# Plot activations histograms

# create a representatives dataframe
representatives_l = []
for digit in range(10):
    representatives_l.append(np.concatenate((nact_representative[digit].iloc[:].values.flatten(), np.array([digit]))))

reprs = np.array(representatives_l)
df_reprs = pd.DataFrame(reprs)

# calculate total mean activations vector
mean_nact_data = df_reprs.iloc[:, :110].mean()

plt.figure(figsize=(15,15))
plt.stem(mean_nact_data[:110])

plt.xlabel('Nueron ID')
plt.ylabel('Relative Frequency')
plt.title('Mean Activations')

plt.savefig(results_dir + '/mean_activations.png')

# mean activation vectors by digit
fig = plt.figure(figsize=(18,18))
ax = fig.subplots(nrows=5, ncols=2)

digit = 0
for row in ax:
    for col in row:
        activations_digit = df_reprs[df_reprs[110] == digit].iloc[:, :110]
        col.stem(activations_digit.values.flatten())
        col.set_title('Digit ' + str(digit))
        digit += 1

plt.savefig(results_dir + '/mean_activations_by_class.png')

# plot activations on window
for i in range(11):
    save_destination = results_dir + '/activations_all_w_' + str(10*i) + '_' + str(10*(i+1)) + '.png'
    plot_activations_on_window(10*i, 10*(i+1), mean_nact_data, df_reprs, save_destination)

# plot combined 1st and 2nd layer
plot_activations_on_window(0, 50, mean_nact_data, df_reprs, results_dir + '/activations_all_1st_layer.png')
plot_activations_on_window(50, 100, mean_nact_data, df_reprs, results_dir + '/activations_all_2nd_layer.png')
