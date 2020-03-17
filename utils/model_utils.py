from __future__ import print_function
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import os
from logger import Logger
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from config import Config
from captum.attr import LayerConductance
import gc


class Scheduler(object):
    """
    An anbstract class representing a scheduler. All other custom schedulers
    should subclass it. All other subclasses should override ``step()``
    """
    def __init__(self):
        self.t = 0 # timestep counter
        self.n_points = 0  # number of scheduler updates

    def f_schedule(self):
        raise NotImplementedError

    def step(self):
        scheduler_value = self.f_schedule()
        self.update_time()
        return scheduler_value

    def update_time(self):
        if self.t < self.n_points - 1:
            self.t += 1
        else:
            # lock value at last timestep
            self.t = self.n_points - 1
    
    def get_prob(self):
        return self.f_schedule()


class LinearScheduler(Scheduler):
    """
    A linear scheduler which is given a `start` and `end` value and draws
    a line between them by interpolating `n_points` between them.
    """
    def __init__(self, point, n_points):
        super(LinearScheduler, self).__init__()
        self.start = point[0]
        self.end = point[1]
        self.n_points = n_points
        self.time = np.linspace(self.start, self.end, self.n_points)
        self.t = 0

    def f_schedule(self, idx=None):
        if idx is None:
            idx = self.t
        return self.time[idx]
        

class MultiplicativeScheduler(Scheduler):
    """
    A scheduler which updates its value by producing a geometric space
    """
    def __init__(self, point, n_points):
        super(MultiplicativeScheduler, self).__init__()
        self.start = point[0]
        if point[0] == 0.0:
            self.start = point[0] + 0.0001
        self.end = point[1]
        self.n_points = n_points
        self.time = np.geomspace(self.start, self.end, self.n_points)
        self.t = 0
    
    def f_schedule(self, idx=None):
        if idx is None:
            idx = self.t
        return self.time[idx]


class ExponentialScheduler(Scheduler):
    """
    A scheduler which exponentially increases is value
    """
    def __init__(self, point, n_points, gamma):
        super(ExponentialScheduler, self).__init__()
        self.start = point[0]    
        self.end = point[1]
        self.n_points = n_points
        self.gamma = gamma
        self.time = np.linspace(0, self.n_points, self.n_points)
        self.exp_t = self.get_function()
        self.t = 0
    
    def get_function(self):
        return (self.end - self.start)*(1 - np.exp(-self.gamma * self.time)) \
               + self.start
    
    def f_schedule(self, idx=None):
        if idx is None:
            idx = self.t
        return self.exp_t[idx]


class StepScheduler(Scheduler):
    """
    A scheduler which holds steady state for a given number of updates 
    and then jumps to the next value.
    """
    def __init__(self, point, n_points):
        super(StepScheduler, self).__init__()
        self.start = point[0]
        self.end = point[1]
        self.n_points = n_points
        self.time = self.start * np.ones(n_points)
        self.t = 0

    def f_schedule(self, idx=None):
        if idx is None:
            idx = self.t
        return self.time[idx]



def get_tensors(only_cuda=False, omit_objs=[]):
    """
    :return: list of active PyTorch tensors
    >>> import torch
    >>> from torch import tensor
    >>> clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> device = torch.device(device)
    >>> only_cuda = True if torch.cuda.is_available() else False
    >>> t1 = tensor([1], device=device)
    >>> a3 = tensor([[1, 2], [3, 4]], device=device)
    >>> # print(get_all_tensor_names())
    >>> tensors = [tensor_obj for tensor_obj in get_tensors(only_cuda=only_cuda)]
    >>> # print(tensors)
    >>> # We doubled each t1, a3 tensors because of the tensors collection.
    >>> expected_tensor_length = 2
    >>> assert len(tensors) == expected_tensor_length, f"Expected length of tensors {expected_tensor_length}, but got {len(tensors)}, the tensors: {tensors}"
    >>> exp_size = (2,2)
    >>> act_size = tensors[1].size()
    >>> assert exp_size == act_size, f"Expected size {exp_size} but got: {act_size}"
    >>> del t1
    >>> del a3
    >>> clean_gc_return = map((lambda obj: del_object(obj)), tensors)
    """
    add_all_tensors = False if only_cuda is True else True
    # To avoid counting the same tensor twice, create a dictionary of tensors,
    # each one identified by its id (the in memory address).
    tensors = {}
    # omit_obj_ids = [id(obj) for obj in omit_objs]
    def add_tensor(obj):
        if torch.is_tensor(obj):
            tensor = obj
        elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
            tensor = obj.data
        else:
            return
        if (only_cuda and tensor.is_cuda) or add_all_tensors:
            tensors[id(tensor)] = tensor
    for obj in gc.get_objects():
        try:
            # Add the obj if it is a tensor.
            add_tensor(obj)
            # Some tensors are "saved & hidden" for the backward pass.
            if hasattr(obj, 'saved_tensors') and (id(obj) not in omit_objs):
                for tensor_obj in obj.saved_tensors:
                    add_tensor(tensor_obj)
        except Exception as ex:
            pass
            # print("Exception: ", ex)
            # logger.debug(f"Exception: {str(ex)}")
    return tensors.values()  # return a list of detected tensors



def train(model, train_loader, optimizer, epoch,
          writer=None,
          attributor=None,
          drop_scheduler=True,
          max_p_drop=None,
          mix_rates=False,
          plain_drop_flag=False,
          p_schedule=None):
    """
    Function that trains the given model for an epoch and returns the 
    respective loss and accuracy after the epoch is over.
    Args:
        model (torch.nn.Module): pytorch model to be trained
        train_loader (torch.utils.data.Dataloader): the train set 
            pytorch iterator
        optimizer (torch.optim.optimizer): optimizer to be used
        epoch (int): epoch id
        writer (torch.utils.tensorboard): SummaryWriter which is used for
            Tensorboard logging
        attributor (list): list of captum.attr instances which is used for
            attributing the importance of a neuron  
        drop_scheduler (bool):
        max_p_drop (float): value at which the drop prob saturates
        mix_rates (bool): handles the use of mixed drop rates
        plain_drop (bool): used for traditional dropour setup
        p_schedule (Scheduler): the scheduler instance which is used
    
    Returns:
        train_loss():
        train_acc():
    """
          
    model.train()
    train_loss = 0
    correct = 0
    step = 0
    #attributor = None  # hach to dismiss previous values

    ###########################################################################
    # drop schedules used
    ###########################################################################
    # for intel drop
    drop_list = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # for plain drop list
    #drop_list = [0.0, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5] # l schedule 0
    #drop_list = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # l schedule 1
    #drop_list = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # l schedule 2
    
    # for mixout
    #drop_list = [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5]
    ###########################################################################


    # if mix_rates:
    #     # (plain_drop, intel_drop)
    #     drop_list = [(max_p_drop - p, p) for p in drop_list]

    # dropout scheduler
    if drop_scheduler:
        if epoch <= len(drop_list):
            p_drop = drop_list[epoch-1]
        else:
            p_drop = drop_list[-1]
    
    if p_schedule is not None:
        p_drop = p_schedule.get_prob()
        print("Using custom scheduler")
        if mix_rates:
            # p_drop = (plain_drop, intel_drop)
            p_drop = (config.p_drop - p_drop, p_drop)
            print(f"added mixed rates --{p_drop[0]}: random-- & --{p_drop[1]}: intel--")

    
    if plain_drop_flag:
        print("Enabling plain dropout")
        p_drop = max_p_drop
    
    model.set_dropout(p_drop, mix_rates)
    print(f"Model is trained with p_drop {model.p_drop}")

    model.reset_drop_cnt()

    #import pdb; pdb.set_trace()

    #print(f"Dropout probability {p_drop}")
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                                    total=len(train_loader)):
        #print(f"batch idx is {batch_idx}")
        data, target = data.to(model.device), target.to(model.device)
        #import pdb; pdb.set_trace()
        #attributor = \
        #        [LayerConductance(model, model.fc[i]) 
        #            for i in range(model.n_fc_layers-1)]

        if p_schedule is not None:
            if batch_idx == 0 and epoch ==1:
                print(f"Using custom scheduler {p_schedule}")
            p_drop = p_schedule.step()
            if mix_rates:
                # p_drop = (plain_drop, intel_drop)
                p_drop = (max_p_drop - p_drop, p_drop)

            model.set_dropout(p_drop, mix_rates)


        if attributor is None:
            # plain dropout case
            # model.init_mask(p_drop=p_drop)
            if p_drop != 0.0:
                if batch_idx == 0: print(f"Model trained with p={p_drop}")
                model.update_mask(p_drop=p_drop)
            else:
                if batch_idx == 0: print(f"zero dropout value")
                model.init_mask(trick="ones")
                #print("no dropout is applied")
        elif attributor is not None:
            # trick to avoid calculating importances when dropout prob is zero
            if plain_drop_flag:
                if batch_idx == 0: print(f"plain dropout is applied with p {p_drop}")
                model.update_mask(importance=None, p_drop=p_drop)
            elif (mix_rates and p_drop[1] == 0.0):
                if batch_idx == 0: print(f"intel mode is off")
                model.update_mask(importance=None, p_drop=p_drop[0])
            elif p_drop == 0.0:
                if batch_idx == 0: print("applying dropout")
                model.init_mask(trick="ones")
            else:
                if batch_idx == 0 and epoch ==1: print(f"entered attribution")
                model.eval()
                neuron_imp = []
                baseline = data*0.0
                for lc in attributor:
                    #print("entered attribution")
                    d1 = torch.clone(data)
                    tmp_tensor = lc.attribute(d1,
                                              baselines=baseline,
                                              target=target,
                                              n_steps=25)
                    
                    #torch.cuda.empty_cache()

                    #neuron_imp.append(torch.sum(lc.attribute(data,
                    #                 baselines=baseline, target=target),
                    #                 dim=0))
                    #print("sumed over batch importances")
                    neuron_imp.append(torch.sum(tmp_tensor, dim=0))
                    #neuron_imp.append(tmp_tensor)
                    
                    #del tmp_tensor
                    
                    #del lc
                    #gc.collect()
                    #torch.cuda.empty_cache()
                    #import pdb; pdb.set_trace()
                model.update_mask(neuron_imp, p_drop, mix_rates)
                model.train()
                #import pdb; pdb.set_trace()
    
        # the following line should be uncommented in case
        # activations are desired
        #_, output = model(data)
        if data.shape[0] != 64:
            print(data.shape)
        output = model(data)
        
        optimizer.zero_grad()
        
        # added for debugging purposes
        #pasok = get_tensors(only_cuda=True)
        #import pdb;  pdb.set_trace()
        
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        
        #######################################################################
        ############ TENSORBOARD LOGGING    ###################################
        #######################################################################
        # this usually is ommited in the train loop
        # however we track it for debugging purposes
        #pred = output.argmax(dim=1, keepdim=True)
        #correct += pred.eq(target.view_as(pred)).sum().item()
        ##import pdb; pdb.set_trace()
        #if (batch_idx+1) % 500 == 0 and (epoch+1) % 4 == 0:
        #    for tag, value in model.named_parameters():
        #        tag = tag.replace(".", "/")
        #        writer.add_histogram(tag,
        #                             value.data.cpu(),
        #                             step, bins="auto")
        #        #writer.add_histogram(tag + '/grad',
        #        #                     value.grad.data.cpu(),
        #        #                     step, bins="auto")
        #        writer.flush()
        #    step += 1
    
    if p_schedule is not None:
        print(f"Scheduler probability is {p_drop}")    

        
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.sampler),
          100. * batch_idx / len(train_loader.sampler), loss.item()))

    train_loss /= len(train_loader)
    train_acc = 100. * correct / len(train_loader.sampler)


    #writer.add_scalars("loss_curves", {"train": train_loss}, epoch-1)
    #writer.add_scalars("accuracy_curve", {"train": train_acc}, epoch-1)
    
    return train_loss, train_acc


def validate(config, model, val_loader, epoch=None, writer=None):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(config.device), target.to(config.device)
            #_, output = model(data)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.sampler)

        #writer.add_scalars("loss_curves", {"val": val_loss}, epoch-1)
        #writer.add_scalars("accuracy_curve", {"val": accuracy}, epoch-1)
        

        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
              val_loss, correct, len(val_loader.sampler), accuracy))

    return val_loss, accuracy


def test(config, model, test_loader):
    model.eval()
    model_name = model.__class__.__name__
    test_loss = 0
    correct = 0
    misclassified_batch = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            #_, output = model(data)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            misclassified_batch.append((target.view_as(pred) - pred).clone().cpu().detach().numpy())

            test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct / len(test_loader.dataset)
        print('{} at Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            model_name, test_loss, correct, len(test_loader.dataset), accuracy))

    misclassified = misclassified_batch[0]
    for misclassified_b in misclassified_batch[1:]:
        misclassified = np.append(misclassified, misclassified_b, axis=0)

    return accuracy, misclassified


def normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def quantize(df, act_threshold, quant_limits):
    x = df.copy().values
    quant_limits.insert(0, act_threshold)
    n_quants = len(quant_limits)
    for i in range(1, n_quants):
        x[(x > quant_limits[i-1]) & (x <= quant_limits[i]) & (x != (i-1)/n_quants)] = i / n_quants
    x[(x > quant_limits[i]) & (x != i/n_quants)] = (i+1) / n_quants
    return pd.DataFrame(x)


def convert_to_binary(df, threshold):
    x = df.copy().values
    x[x > threshold] = 1
    x[x <= threshold] = 0
    return pd.DataFrame(x)


def get_normalized_cont_data(train_data_file, test_data_file, n_features, act_threshold):
    df_train = pd.read_csv(train_data_file, header=None)
    df_test = pd.read_csv(test_data_file, header=None)

    y_train = df_train.iloc[:, n_features]
    X_train = df_train.iloc[:, :n_features]
    y_test = df_test.iloc[:, n_features]
    X_test = df_test.iloc[:, :n_features]

    # apply ReLU on the continuous activations data
    # X_train_relu = X_train.where(X_train > 0, 0)
    # X_test_relu = X_test.where(X_test > 0, 0)

    # normalize data - convert values to the range (0,1)
    # X_train_n = normalize(X_train_relu)
    # X_test_n = normalize(X_test_relu)

    X_train_n = normalize(X_train)
    X_test_n = normalize(X_test)

    # apply a larger ReLU threshold: act_threshold
    X_train_relu_act = X_train_n.where(X_train_n > act_threshold, 0)
    X_test_relu_act = X_test_n.where(X_test_n > act_threshold, 0)

    return X_train_relu_act, X_test_relu_act, y_train, y_test


def convert_data_to_binary(train_data_file, test_data_file, n_features, act_threshold):
    X_train_relu_act, X_test_relu_act, y_train, y_test = get_normalized_cont_data(train_data_file, test_data_file, n_features, act_threshold)

    # convert data to multihot/binary vectors
    X_train_b_act = convert_to_binary(X_train_relu_act, 0)
    X_test_b_act = convert_to_binary(X_test_relu_act, 0)

    return X_train_b_act, X_test_b_act, y_train, y_test


def convert_data_to_quantized(train_data_file, test_data_file, n_features, act_threshold, quant_limits):
    X_train_relu_act, X_test_relu_act, y_train, y_test = get_normalized_cont_data(train_data_file, test_data_file, n_features, act_threshold)

    # quantize data
    X_train_q = quantize(X_train_relu_act, act_threshold, quant_limits)
    X_test_q = quantize(X_test_relu_act, act_threshold, quant_limits)

    return X_train_q, X_test_q, y_train, y_test


def prepare_experiment_logs(current_dir, current_filename, specific_identifier):
    # create results dir if doesn't exist
    results_dir = current_dir + '/results/' + current_filename
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = results_dir + '/results' + specific_identifier + '.txt'
    # escape overwriting the same file by creating a new one
    i = 1
    while os.path.exists(results_file):
        specific_identifier_ = specific_identifier + '_' + str(i)
        results_file = results_dir + '/results' + specific_identifier_ + '.txt'
        i += 1

    # redirect to results file
    sys.stdout = Logger(results_file)

    return results_dir, results_file


def get_dataframes_by_option(option, train_data_file, test_data_file, config, act_threshold=0, quant_limits=None):
    n_features = config.n_acts
    if option == 'binary':
        X_train, X_test, y_train, y_test = convert_data_to_binary(train_data_file, test_data_file, n_features, act_threshold)
    elif option == 'quantized':
        X_train, X_test, y_train, y_test = convert_data_to_quantized(train_data_file, test_data_file, n_features, act_threshold, quant_limits)
    elif option == 'cont_normalized_threshold':
        X_train, X_test, y_train, y_test = get_normalized_cont_data(train_data_file, test_data_file, n_features, act_threshold)
    elif option == 'continuous_raw':
        df_train = pd.read_csv(train_data_file, header=None)
        df_test = pd.read_csv(test_data_file, header=None)
        y_train = df_train.iloc[:, n_features]
        X_train = df_train.iloc[:, :n_features]
        y_test = df_test.iloc[:, n_features]
        X_test = df_test.iloc[:, :n_features]
    return X_train, X_test, y_train, y_test


def get_binary_representatives(train_data_file, test_data_file, n_features, act_threshold):
    nact_representative = get_representatives(train_data_file, test_data_file, n_features, act_threshold)
    # convert each representative to a binary vector
    nact_representative_b = {}

    for digit in range(10):
        nact_representative_b[digit] = convert_to_binary(nact_representative[digit], 0.5)

    return nact_representative_b


def get_representatives(train_data_file, test_data_file, n_features, act_threshold):
    # convert data to binary vectors
    X_train, X_test, y_train, y_test = convert_data_to_binary(train_data_file, test_data_file, n_features, act_threshold)
    # combine data to dataframe
    nact_data = pd.concat([X_train, y_train], axis=1)

    # create a dict to store the mean vector of each class
    nact_representative = {}
    for digit in range(10):
        nact_representative[digit] = nact_data[nact_data[n_features] == digit].iloc[:, :n_features].mean()

    return nact_representative

    # convert each representative to a binary vector
    nact_representative_b = {}

    for digit in range(10):
        nact_representative_b[digit] = convert_to_binary(nact_representative[digit], 0.5)

    return nact_representative_b


def visualize_activations(config, results_dir, train_data_file, test_data_file, act_threshold):
    # TODO implementation for more than 2 hidden layers
    if len(config.layers) > 3:
        print('Not yet implemented for more than 2 hidden layers')
        sys.exit()

    height = max(config.layers) + 20
    width = 80
    visualization = np.zeros((height, width))

    n_features = config.n_acts
    nact_representative_b = get_binary_representatives(train_data_file, test_data_file, n_features, act_threshold)

    step = 0
    for digit in range(10):
        visualization[10:(config.layers[0]+10), 5 + digit + step] = nact_representative_b[digit].iloc[
                                                                    0:config.layers[0]].values.flatten() * (digit + 1)
        visualization[10:(config.layers[1]+10), 30 + digit + step] = nact_representative_b[digit].iloc[
                                                                config.layers[0]:(config.layers[0] + config.layers[1])
                                                                ].values.flatten() * (digit + 1)
        visualization[(height//2 - 5):(height//2 + 5), 55 + digit + step] = nact_representative_b[digit].iloc[
                                                                          (n_features-10):n_features
                                                                          ].values.flatten() * (digit + 1)
        step += 1

    plt.figure(figsize=(18, round(18*height/width)))
    plt.imshow(visualization, cmap='gist_stern')
    plt.gca().set_title('Digits Activations')

    colorbar = plt.colorbar(boundaries=np.arange(0.5, 11.5, 1), orientation='horizontal', fraction=0.082, pad=0.0)
    # colorbar labels
    labels = np.arange(0, 10, 1)
    loc = labels + 1
    colorbar.set_ticks(loc)
    colorbar.set_ticklabels(labels)

    vis_file = results_dir + '/digits_activations_' + str(n_features) + '.png'
    plt.savefig(vis_file)
    print('Visualization saved at: {}'.format(vis_file))


class EuclideanClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, train_data_file, test_data_file, n_acts, act_threshold):
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.n_acts = n_acts
        self.act_threshold = act_threshold
        self.representatives = None
        self.X_mean = None

    def fit(self, ignore_last_layer=False):
        self.representatives = get_representatives(self.train_data_file, self.test_data_file, self.n_acts, self.act_threshold)
        if ignore_last_layer:
            for i in range(10):
                self.representatives[i] = self.representatives[i][:self.n_acts-10]
        self.X_mean = []
        for i in range(10):
            self.X_mean.append(self.representatives[i])

        self.X_mean = np.array(self.X_mean)
        return self

    def predict(self, X):
        closest = np.argmin(euclidean_distances(X, self.X_mean), axis=1)
        return closest

    def score(self, X, y):
        y_ = self.predict(X)
        return np.sum(y == y_) / y.shape[0]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, config=None, delta=0, model_id=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_id = model_id
        if config == None:
            self.config = Config()
        else:
            self.config = config

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # save model
        if self.config.save_model:
            torch.save(model.state_dict(), self.config.saved_model_path)

        self.val_loss_min = val_loss
