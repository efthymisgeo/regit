from __future__ import print_function
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import os
#from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from configs.config import Config
from captum.attr import LayerConductance
import gc


class Scheduler(object):
    """
    An anbstract class representing a scheduler. All other custom schedulers
    should subclass it. All other subclasses should override ``step()``
    """
    def __init__(self):
        self.t = 0  # timestep counter
        self.n_points = 0  # number of scheduler updates
        self.t_osc = 0  # periodic osc timestep counter
        self.f_osc = 0
        self.delay = 0

    def f_schedule(self):
        raise NotImplementedError

    def step(self):
        scheduler_value = self.f_schedule()
        self.update_time()
        return scheduler_value

    def update_time(self):
        """
        Function which updates time for both the schduler and the oscillation
        """
        # update scheduler timescale
        if self.t < self.n_points - 1:
            self.t += 1
        else:
            # lock value at last timestep
            self.t = self.n_points - 1
        
        # update oscillation timescale
        if self.f_osc != 0 and self.t >= self.delay:
            self.t_osc = (self.t_osc + 1) % self.f_osc
    
    def get_prob(self):
        return self.f_schedule()


class LinearScheduler(Scheduler):
    """
    A linear scheduler which is given a `start` and `end` value and draws
    a line between them by interpolating `n_points` between them. Moreover the
    class offers the ability of additing oscillating noise in the scheduler.
    """
    def __init__(self, point, n_points, delay=0.0, eps=0.000001,
                 f_osc=0, a_osc=0.0):
        super(LinearScheduler, self).__init__()
        self.start = point[0]
        self.end = point[1]
        self.n_points = n_points
        self.eps = eps
        self.delay = delay
        self.f_osc = f_osc
        self.a_osc = a_osc
        self.time = self.add_delay()
        if a_osc != 0:
            self.sin_osc = self.add_sinus()        
        #self.t = 0
    
    def add_delay(self):
        """function which adds delay"""
        if self.delay > 0.0:
            pad = np.zeros(self.delay)
            if self.n_points - self.delay <= 0:
                raise ValueError("Delay exceeds total number of points")
            else:
                time = self._make_line(self.start + self.eps,
                                       self.end,
                                       self.n_points - self.delay)
            time = np.concatenate((pad, time), axis=0)
        else:
            time = self._make_line(self.start + self.eps,
                                   self.end,
                                   self.n_points)
        return time
    
    
    @staticmethod
    def _make_line(p_start, p_end, points):
        """Function which contructs a line from starting point p_start and ends
        at ending point p_end. The line is essentially an interpolation of
        `points` inbetween the starting and the ending point.
        """
        return np.linspace(p_start, p_end, points)
    
    def add_sinus(self):
        """
        function which adds sinusodial oscillation with frequency f_osc and
        amplitude a_osc on top of given curve. To avoid negative prob values 
        we have taken the absolute value of the overall curve.
        """
        # sample f_osc points from a sinus with amp a_osc
        return self.a_osc * np.sin(np.linspace(0, 2*np.pi, self.f_osc))


    def f_schedule(self, idx=None):
        if idx is None:
            idx = self.t
        out = self.time[idx]
        if self.a_osc != 0 and self.t >= self.delay:
            out = np.abs(out + self.sin_osc[self.t_osc])
        return out
        

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


class PowerScheduler(Scheduler):
    """
    A scheduler which increases value based on some power
    """
    def __init__(self, point, n_points, gamma):
        super(PowerScheduler, self).__init__()
        self.start = point[0]    
        self.end = point[1]
        self.n_points = n_points
        self.gamma = gamma
        self.time = np.linspace(0, self.n_points, self.n_points)
        self.exp_t = self.get_function()
        self.t = 0
    
    def get_function(self):
        return (self.end - self.start)*(np.exp(self.gamma * self.time) - 1)
    
    def f_schedule(self, idx=None):
        if idx is None:
            idx = self.t
        return self.exp_t[idx]


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
    def __init__(self, point, n_points, step_freq=10):
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


def sample_tensor(x, p_x):
    """
    Function which samples a proportion p_x of a given tensor x along its 0 dim
    For instance for a [10, 33]  tensor it will return a [5, 33] tensor
    Args:
        x (torch.tensor): B x D
        p_x (float): float in range [0,1]
    Output:
        sampled_x (torch.tensor): int(B*p_x) x D
    """
    if p_x <= 0.0 or p_x >= 1.0:
        raise ValueError("Proportion not in valid range.")
    s_dim = x.size(0)
    n_keep = int(s_dim * p_x)
    rand_x = torch.randperm(s_dim)
    sampled_x = rand_x[:n_keep]
    return sampled_x            


def train(model,
          train_loader,
          optimizer,
          epoch,
          regularization=True,
          writer=None,
          attributor=None,
          drop_scheduler=True,
          max_p_drop=None,
          mix_rates=False,
          plain_drop_flag=False,
          p_schedule=None,
          use_inverted_strategy=True,
          inverted_strategy="importance",
          reset_counter=False,
          sampling_imp=50,
          n_steps=25,
          aggregate=True,
          sample_batch=None,
          sigma_attr=None,
          sigma_input=None,
          adapt_to_tensor=False,
          momentum=None,
          per_sample_noise=False):
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
        max_p_drop (float): value at which the drop prob saturates
        mix_rates (bool): handles the use of mixed drop rates
        plain_drop (bool): used for traditional dropour setup
        p_schedule (Scheduler): the scheduler instance which is used
        use_inverted_strategy (bool): handles the use of inverted drop strategy
        inverted_strategy (str): specifies the strategy to be used
        reset_counter (bool): reset or not the switch counter at every epoch
        sampling_imp (list/int): int or list of ints which indicate the number
            of batches a single mask will be applied. In other words, indicates
            the number of epochs that will be through until the next importance
            calculation.
        n_steps (int): number of interpolation steps
        aggregate (bool): a boolean variable which indicates if the importances
            will be aggregated over the batch or not (one mask/multiple masks)
        sample_batch (float): a float which specifies the amount of batch that
            will be held for calculating the attribution
        sigma_attr (float): a float which specifies the std of noise to be
            added on top of the attribution
        sigma_input (float): a float which specifies the std of noise to be
            added on the input to "mislead" the attributions
        momentum (float): momentum term to be added in the attribution
            calculation
        per_sample_noise (bool): when true the noise is added per sample rather
            than per batch, enforcing the use of a different mask for every
            sample
    Returns:
        train_loss (float): list of train losses per epoch
        train_acc (float): list of train accuracies per epoch
        prob_value (float): list of dropout probabilities
    """
    ###########################################################################
    #### experimental mode: probabilistic, COUNT_SWITCHES
    ###########################################################################
    probabilistic = False
    COUNT_SWITCHES = False  # handles the use or not of the `update_sw_stats`
    ###########################################################################
    #### tested mode
    ###########################################################################
    USE_INVERTED_DROP_STRATEGY = use_inverted_strategy      
    INVERTED_DROP_STRATEGY = inverted_strategy
    RESET_COUNTER = reset_counter
    SAMPLE_BATCH = False
    P_BATCH = 0.5      
    model.train()
    train_loss = 0
    correct = 0
    batch_loss = []
    step = 0
    prob_value = 0

    SAMPLING = sampling_imp # sampling masks in a minibatch (aka calculating attributions)
    if isinstance(sampling_imp, list):
        if epoch <= len(sampling_imp):
            SAMPLING = sampling_imp[epoch-1]
        else:
            SAMPLING = sampling_imp[-1]
    else:
        SAMPLING = sampling_imp

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
        # TODO investigate whether to use get_prob() or step()
        p_drop = p_schedule.get_prob()
        #p_drop = p_schedule.step()
        print("Using custom scheduler")
        if mix_rates:
            # p_drop = (plain_drop, intel_drop)
            p_drop = (max_p_drop - p_drop, p_drop)
            print(f"added mixed rates --{p_drop[0]}: random-- & --{p_drop[1]}: intel--")

    
    if plain_drop_flag:
        print("Enabling plain dropout")
        p_drop = max_p_drop
    
    if not regularization:
        # overwrite existing values for proper use
        p_drop = 0.0
        mix_rates = False
        p_schedule = None
        attributor = None

    model.set_dropout(p_drop, mix_rates)
    print(f"Model is trained with p_drop {model.p_drop}")
    
    if RESET_COUNTER:
        model.reset_drop_cnt()

    #import pdb; pdb.set_trace()

    #print(f"Dropout probability {p_drop}")
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                                    total=len(train_loader)):
        data, target = data.to(model.device), target.to(model.device)
        batch_size = data.size(0)
        
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
                model.update_mask(p_drop=p_drop, aggregate=aggregate,
                                  batch_size=batch_size)
                model.update_inv_drop_factor(strategy="bernoulli")
            else:
                if batch_idx == 0: print(f"zero dropout value. no drop applied")
                model.init_mask(trick="ones")
        elif attributor is not None:
            # trick to avoid calculating importances when dropout prob is zero
            if plain_drop_flag:
                if batch_idx == 0: print(f"plain dropout is applied with p {p_drop}")
                model.update_mask(importance=None, p_drop=p_drop)
            elif (mix_rates and p_drop[1] == 0.0):
                if batch_idx == 0: print(f"intel mode is off")
                model.update_mask(importance=None, p_drop=p_drop[0])
            elif p_drop == 0.0:
                if batch_idx == 0: print("applying unitary mask instead of dropout")
                model.init_mask(trick="ones")
                neuron_imp = None  # this is for compatibility in sampling case
            else:
                model.eval()
                if batch_idx % SAMPLING == 0:
                    #print("ENTERED ATTRIBUTION CALCULATION AREA")
                    neuron_imp = []
                    baseline = data*0.0
                    for lc in attributor:
                        #print("entered attribution")
                        if SAMPLE_BATCH:
                            dl = sample_tensor(data, P_BATCH)
                        else:
                            d1 = torch.clone(data)
                        
                        
                        tmp_tensor = \
                            lc.attribute_noise(d1,
                                               baselines=baseline,
                                               target=target,
                                               n_steps=n_steps,
                                               sample_batch=sample_batch,
                                               sigma_attr=sigma_attr,
                                               sigma_input=sigma_input,
                                               adapt_to_tensor=adapt_to_tensor,
                                               momentum=momentum,
                                               aggregate=aggregate,
                                               per_sample_noise=per_sample_noise)
                        
                        #torch.cuda.empty_cache()

                        #neuron_imp.append(torch.sum(lc.attribute(data,
                        #                 baselines=baseline, target=target),
                        #                 dim=0))
                        
                        
                        ######################################################
                        #####   UNCOMMENT FOLLOWING LINES for neuron_imp
                        ######################################################
                        #print("sumed over batch importances")
                        #if aggregate:
                        #    neuron_imp.append(torch.sum(tmp_tensor, dim=0))
                        #else:
                        #    neuron_imp.append(tmp_tensor)
                        neuron_imp.append(tmp_tensor)
                        
                        #del tmp_tensor
                        
                        #del lc
                        #gc.collect()
                        #torch.cuda.empty_cache()
                        #import pdb; pdb.set_trace()
                

                ###############################################################
                #### probabilistic is at experimental mode
                ###############################################################
                if batch_idx == 3 and probabilistic:
                    model._plot_importance(neuron_imp, epoch, batch_idx)
                
                model.update_mask(neuron_imp,
                                  p_drop,
                                  mix_rates,
                                  aggregate,
                                  batch_size,
                                  probabilistic=probabilistic)
                
                if COUNT_SWITCHES:
                    model.update_sw_stats()
                
                # added for debugging purposes
                if (batch_idx == 3) and probabilistic:
                    model._count_prob_switches()
                
                if USE_INVERTED_DROP_STRATEGY:
                    model.update_inv_drop_factor(strategy=INVERTED_DROP_STRATEGY)
                model.train()
                #import pdb; pdb.set_trace()
    
        # the following line should be uncommented in case
        # activations are desired
        #_, output = model(data)
        #if data.shape[0] != 64:
        #    print(data.shape)
        output = model(data)
        
        optimizer.zero_grad()
        
        # added for debugging purposes
        #pasok = get_tensors(only_cuda=True)
        #import pdb;  pdb.set_trace()
        
        loss = F.nll_loss(output, target)
        batch_loss.append(loss.item())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        prob_value = p_drop

        # if epoch==1 and batch_idx == 500:
        #     import pdb; pdb.set_trace()
        #     print(model.switch_counter) 

        #######################################################################
        ############ TENSORBOARD LOGGING    ###################################
        #######################################################################
        # this usually is ommited in the train loop
        # however we track it for debugging purposes
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
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
    
    return train_loss, train_acc, prob_value, batch_loss


def validate(model,
             val_loader,
             epoch=None,
             writer=None):
    model.eval()
    val_loss = 0
    batch_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(model.device), target.to(model.device)
            #_, output = model(data)
            output = model(data)
            loss = F.nll_loss(output, target)
            batch_loss.append(loss.item())
            val_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.sampler)

        #writer.add_scalars("loss_curves", {"val": val_loss}, epoch-1)
        #writer.add_scalars("accuracy_curve", {"val": accuracy}, epoch-1)
        

        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
              val_loss, correct, len(val_loader.sampler), accuracy))

    return val_loss, accuracy, batch_loss


def test(model, test_loader):
    model.eval()
    model_name = model.__class__.__name__
    test_loss = 0
    correct = 0
    batch_loss = []
    misclassified_batch = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model.device), target.to(model.device)
            #_, output = model(data)
            output = model(data)
            loss = F.nll_loss(output, target)
            batch_loss.append(loss.item())
            test_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            misclassified_batch.append((target.view_as(pred) - pred).clone().cpu().detach().numpy())

        test_loss /= len(test_loader)

        accuracy = 100. * correct / len(test_loader.dataset)
        print('{} at Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            model_name, test_loss, correct, len(test_loader.dataset), accuracy))

    misclassified = misclassified_batch[0]
    for misclassified_b in misclassified_batch[1:]:
        misclassified = np.append(misclassified, misclassified_b, axis=0)

    return test_loss, accuracy, misclassified, batch_loss



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, save_model=True, delta=0, ckpt_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation 
                loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation 
                loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify 
                as an improvement. Default: 0
            ckpt_path (str): the path under which the model will be stored
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.save_model = save_model
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ckpt_path = ckpt_path
        self.best_epoch_id = 0

    def __call__(self, val_loss, model):

        score = -val_loss
        self.best_epoch_id += 1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.best_epoch_id = self.best_epoch_id - self.patience
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print("Validation loss decreased "
                  f"{self.val_loss_min:.6f} --> {val_loss:.6f}." 
                  "Saving model ...")

        # save model
        if self.save_model:
            torch.save(model.state_dict(),
                       self.ckpt_path + ".pt")

        self.val_loss_min = val_loss

    def reset_counter(self):
        """re-init counter to 0
        """
        self.counter = 0

