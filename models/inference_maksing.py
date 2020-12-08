import os
import sys
import csv
from tqdm import tqdm
import json
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.models as torch_models
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from modules.vgg import *
from modules.models import CNN2D, CNNFC
from utils.model_utils import train, test, validate, EarlyStopping
from utils.mnist import MNIST, CIFAR10, CIFAR100, IMAGE_NET, STL10, SVHN, IMAGE_NET
from utils.config_loader import *
from utils.opts import load_experiment_options
import matplotlib.pyplot as plt
from models.end2end import run_training
from captum.attr import LayerConductance
import matplotlib.pyplot as plt
from utils.importance import Importance, Attributor


def flip_tensor(tensor):
    inv_idx = torch.arange(tensor.size(0)-1, -1, -1).long()
    inv_tensor = tensor.index_select(0, inv_idx)
    return inv_tensor


def create_mask(n_units,
                keep_perc,
                keep_low_cond=False):
    keep_units = round(n_units * keep_perc)
    prune_units = n_units - keep_units
    mask = torch.ones(n_units)
    mask[keep_units:] = 0.0
    if not keep_low_cond:
        mask = flip_tensor(mask) 

    return mask

def sort_mapping(tensoras):
    tensor_sort, tensor_idx = tensoras.sort()
    _, map_idx = tensor_idx.sort()
    return tensor_sort,  map_idx


class Toy_FC(nn.Module):
    def __init__(self,
                 masking="random",
                 percentage=0.1,
                 n_units=1024,
                 update_mask=False,
                 device="cuda",
                 use_mask=True):
        super(Toy_FC, self).__init__()
        self.n_units = n_units
        self.use_mask = use_mask
        self.fc_0 = nn.Linear(28*28, self.n_units),
        self.activ = nn.ReLU(True)
        self.fc_1 = nn.Linear(self.n_units, 10)

        self.update_mask = update_mask
        self.device = device
        self.masking = masking
        self.percentage = percentage

        self.use_rank = False

        self.mask = self.get_mask()

    def get_mask(self, rankings=None):
        if self.masking == "random" or (rankings is None):
            # import pdb; pdb.set_trace()
            mask = torch.ones(self.n_units) * self.percentage
            mask = torch.bernoulli(mask)
            # print(mask[:15])
        elif self.masking == "cond":
            _, sort_map = sort_mapping(rankings)
            mask = create_mask(self.n_units, self.percentage)
            mask = mask[sort_map]
        else:
            pass
        return mask.to(self.device)

    def reset_mask(self, rankings=None):
        self.use_rank = True
        self.mask = self.get_mask(rankings=rankings)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc_0(x)
        x = self.activ(x)
        if self.use_rank:
            if self.use_mask and self.update_mask:
                mask = self.get_mask()
                x = x * mask
            elif self.use_mask:
                mask = self.mask
                x = x * mask
            else:
                pass
        out = self.fc_1(x)
        return out


ATTRIBUTE_SETUP = {
    "sample_batch": None,
    "n_steps": 15
}

def load_model(saved_model,
               checkpoint_path,
               masking,
               percentage,
               n_units,
               update_mask,
               device,
               use_mask):

    # load model
    saved_model.load_state_dict(torch.load(checkpoint_path,
                                           map_location='cpu'))

    model = Toy_FC(masking=masking,
                   percentage=percentage,
                   n_units=n_units,
                   update_mask=update_mask,
                   device=device,
                   use_mask=use_mask)
    model.fc_0 = saved_model.fc[0]
    model.fc_1 = saved_model.fc[1]
    return model

def attribute_loop(model,
                   data_loader,
                   attributor,
                   n_steps,
                   sample_batch,
                   sigma_attr,
                   sigma_input,
                   adapt_to_tensor,
                   momentum,
                   aggregate,
                   per_sample_noise,
                   respect_attr,
                   device):
    mean_conductance = torch.zeros(1024).to(device)
    rankings = None
    model.eval()
    for batch_idx, (data, target) in tqdm(enumerate(data_loader),
                                                    total=len(data_loader)):
        #import pdb; pdb.set_trace()
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        # print(target.shape)
        batch_size = data.size(0)
        baseline = 0.0 * data
        rankings, statistics, total_conductance, per_class_cond = \
            attributor.get_attributions(data,
                                     baselines=baseline,
                                     target=target,
                                     n_steps=n_steps,
                                     sample_batch=sample_batch,
                                     sigma_attr=sigma_attr,
                                     sigma_input=sigma_input,
                                     adapt_to_tensor=adapt_to_tensor,
                                     momentum=momentum,
                                     aggregate=aggregate,
                                     per_sample_noise=per_sample_noise,
                                     respect_attr=respect_attr,
                                     batch_idx=batch_idx,
                                     calc_stats=False)
        mean_conductance = mean_conductance + rankings[0]
        del rankings
        del total_conductance
        torch.cuda.empty_cache()

    return mean_conductance / batch_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model_path1", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("-mask", "--masking", type=str,
                        choices=["random", "cond"],
                        default="random",
                        help="random or cond masking")
    parser.add_argument("-p", "--percentage", type=float, default=0.1,
                        help="percentage of kept units")
    parser.add_argument("-run", "--n_runs", type=int, default=10,
                        help="the number of inference runs")
    parser.add_argument("--update_mask", action="store_true",
                        help="Flag which handles the update of masks in every"
                        "batch of the inference step. When False a unique mask"
                        "is appied")
    parser.add_argument("--no_mask", action="store_true",
                        help="no masking is applied when this is called")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()
    use_mask = not(args.no_mask)
    print(f"Update mask is {args.update_mask} and no_mask flag is {args.no_mask}")

    # parser.add_argument("-m2", "--model_path2", required=False,
    #                     help="Path to model checkpoint")
    # parser.add_argument("-m3", "--model_path3", required=False,
    #                     help="Path to model checkpoint")
    # load checkpoint
    
    # checkpoint_path = args.model_path
    # # load configs
    data_setup = load_config("configs/dataset/mnist.json")
    model_setup = load_config("configs/model/mnist_toy.json")
    exp_setup = load_config("configs/experiment/mnist_toy.json")
    exp_setup["kwargs"] = get_kwargs(exp_setup)
    exp_setup["device"] = args.device
    # train test split
    data = MNIST(data_setup, exp_setup)
    train_loader = data.get_train_loader()
    test_loader = data.get_test_loader()
    # preliminaries\
    device = exp_setup["device"]
    regularization = exp_setup["regularization"]
    fc_only = exp_setup.get("fc_only", False)
    importance = exp_setup["importance"]  # True to use importance
    use_drop_schedule = True if exp_setup["use_drop_schedule"] is not None else False  # True to use scheduler
    mixout = exp_setup["mixout"]
    plain_drop_flag = exp_setup["plain_drop"]
    if exp_setup["use_drop_schedule"] != {}: 
        custom_scheduler = exp_setup["use_drop_schedule"]["prob_scheduler"]
        if custom_scheduler == "Exp":
            gamma = exp_setup["use_drop_schedule"]["gamma"]
    
    if exp_setup["idrop"] != {}:
        map_rank_method = exp_setup["idrop"].get("method", "bucket")
        p_buckets = exp_setup["idrop"].get("p_buckets", [0.2, 0.8])
        inv_trick = exp_setup["idrop"].get("inv_trick", "dropout")
        alpha = exp_setup["idrop"].get("alpha", 1e-5)
        drop_low = exp_setup["idrop"].get("drop_low", True)
        sigma_drop = exp_setup["idrop"].get("sigma_drop", 0.05)
        rk_history = exp_setup["idrop"].get("rk_history", "short")
    else:
        map_rank_method = False
        p_buckets = [0.9, 0.1]
        inv_trick = "dropout"
        drop_low = True
        alpha = 1e-5
        sigma_drop = 0.05
        rk_history = "short"
    # loss function
    criterion = nn.CrossEntropyLoss()
    # model setup
    fc_setup = model_setup["FC"]
    input_shape = data_setup["input_shape"]
    
    # test_loss, test_acc, _, _ = test(model, test_loader, criterion, device)
    # print(f"The accuracy of the loaded checkpoint is {test_acc}")
    saved_model_1 = CNNFC(input_shape=input_shape,
                    regularization=False,
                    activation=fc_setup["activation"],
                    fc_layers=fc_setup["fc_layers"],
                    add_dropout=fc_setup["fc_drop"],
                    p_drop=fc_setup["p_drop"],
                    drop_low=drop_low,
                    idrop_method=map_rank_method,
                    p_buckets=p_buckets,
                    inv_trick=inv_trick,
                    sigma_drop=sigma_drop,
                    alpha=alpha,
                    rk_history=rk_history,
                    pytorch_dropout=exp_setup["plain_drop"],
                    prior=exp_setup["prior"],
                    device=device,
                    fc_only=fc_only)
    
    # saved_model_2 = CNNFC(input_shape=input_shape,
    #                 regularization=False,
    #                 activation=fc_setup["activation"],
    #                 fc_layers=fc_setup["fc_layers"],
    #                 add_dropout=fc_setup["fc_drop"],
    #                 p_drop=fc_setup["p_drop"],
    #                 drop_low=drop_low,
    #                 idrop_method=map_rank_method,
    #                 p_buckets=p_buckets,
    #                 inv_trick=inv_trick,
    #                 sigma_drop=sigma_drop,
    #                 alpha=alpha,
    #                 rk_history=rk_history,
    #                 pytorch_dropout=exp_setup["plain_drop"],
    #                 prior=exp_setup["prior"],
    #                 device=device,
    #                 fc_only=fc_only)

    # saved_model_3 = CNNFC(input_shape=input_shape,
    #                 regularization=False,
    #                 activation=fc_setup["activation"],
    #                 fc_layers=fc_setup["fc_layers"],
    #                 add_dropout=fc_setup["fc_drop"],
    #                 p_drop=fc_setup["p_drop"],
    #                 drop_low=drop_low,
    #                 idrop_method=map_rank_method,
    #                 p_buckets=p_buckets,
    #                 inv_trick=inv_trick,
    #                 sigma_drop=sigma_drop,
    #                 alpha=alpha,
    #                 rk_history=rk_history,
    #                 pytorch_dropout=exp_setup["plain_drop"],
    #                 prior=exp_setup["prior"],
    #                 device=device,
    #                 fc_only=fc_only)

    aggregate = True
    sample_batch = ATTRIBUTE_SETUP.get("sample_batch", None)
    n_steps = ATTRIBUTE_SETUP.get("n_steps", 5)
    sigma_attr = ATTRIBUTE_SETUP.get("sigma_attr", None)
    sigma_input = ATTRIBUTE_SETUP.get("sigma_input", None)
    momentum = ATTRIBUTE_SETUP.get("momentum", None)
    adapt_to_tensor = ATTRIBUTE_SETUP.get("adapt_to_tensor", False)
    per_sample_noise = ATTRIBUTE_SETUP.get("per_sample_noise", False)
    respect_attr = ATTRIBUTE_SETUP.get("respect_attr", False)
    

    n_runs = args.n_runs
    print(args.model_path1)
    # print(args.model_path2)
    # print(args.model_path3)
    model_1 = load_model(saved_model_1,
                        args.model_path1,
                        args.masking,
                        args.percentage,
                        1024,
                        args.update_mask,
                        device,
                        use_mask)
    # model_2 = load_model(saved_model_2, args.model_path2)
    # model_3 = load_model(saved_model_3, args.model_path3)
    model_1.to(device)

    # mean conductance calculation
    data_loader = train_loader
    y_attr_1 = Attributor(model_1, [model_1.fc_0])
    mean_cond_1 = attribute_loop(model_1,
                                 data_loader,
                                 y_attr_1,
                                 n_steps,
                                 sample_batch,
                                 sigma_attr,
                                 sigma_input,
                                 adapt_to_tensor,
                                 momentum,
                                 aggregate,
                                 per_sample_noise,
                                 respect_attr,
                                 device)
    # import pdb; pdb.set_trace()

    sorted_mean_cond_1, _ = mean_cond_1.sort() 
    x_axis = np.arange(1024)
    fig, ax = plt.subplots()
    plt.plot(x_axis, sorted_mean_cond_1.detach().cpu().numpy(), label=f"Y-Drop")
    plt.legend(loc='upper left')
    plt.savefig("tets-101.png")
    plt.close()


    acc = []
    for k in range(n_runs):
        if args.masking == "random":
            model_1.reset_mask() # update mask in every run
        else:
            model_1.reset_mask(rankings=mean_cond_1)
        _, test_acc_1, _, _ = \
            test(model_1, test_loader, criterion, device)
        print(f"The accuracy of the loaded checkpoint is {test_acc_1}")
        acc.append(test_acc_1)

    if args.update_mask:
        s = "unique mask"
    else:
        s = "updated mask"
    
    print(f"The mean accuracy for {args.masking} masking and {s} \n with "
          f"{args.percentage} remaining neurons neurons is: \n"
          f"{np.mean(acc):.3f} with std {np.std(acc):.3f} over {n_runs} runs.")

        


    
    # y_attr_2 = Attributor(model_2, [model_2.fc[0]])
    # y_attr_3 = Attributor(model_3, [model_3.fc[0]])

    
    
    
    # _, test_acc_1, _, _ = test(model_1, test_loader, criterion, device)
    # # _, test_acc_2, _, _ = test(model_2, test_loader, criterion, device)
    # # _, test_acc_3, _, _ = test(model_3, test_loader, criterion, device)
    # # print(f"The accuracy of the loaded checkpoint is {test_acc_1} and {test_acc_2} and {test_acc_3}")
    # print(f"The accuracy of the loaded checkpoint is {test_acc_1}")
    # calculate mean of mean conductance
    # data_loader = test_loader


    
    
    
    
    
    
    
    
    # mean_cond_1 = attribute_loop(model_1,
    #                              data_loader,
    #                              y_attr_1,
    #                             n_steps,
    #                             sample_batch,
    #                             sigma_attr,
    #                             sigma_input,
    #                             adapt_to_tensor,
    #                             momentum,
    #                             aggregate,
    #                             per_sample_noise,
    #                             respect_attr,
    #                             device)
    # # del model_1
    # # torch.cuda.empty_cache()
    # # model_2.to("cuda:1")
    # # mean_cond_2 = attribute_loop(model_2,
    # #                                 data_loader,
    # #                                 y_attr_2,
    # #                                 n_steps,
    # #                                 sample_batch,
    # #                                 sigma_attr,
    # #                                 sigma_input,
    # #                                 adapt_to_tensor,
    # #                                 momentum,
    # #                                 aggregate,
    # #                                 per_sample_noise,
    # #                                 respect_attr,
    # #                                 device)
    # # # del model_2
    # # # torch.cuda.empty_cache()
    # # # model_3.to("cuda:2")
    # # mean_cond_3 = attribute_loop(model_3,
    # #                                 train_loader,
    # #                                 y_attr_3,
    # #                                 n_steps,
    # #                                 sample_batch,
    # #                                 sigma_attr,
    # #                                 sigma_input,
    # #                                 adapt_to_tensor,
    # #                                 momentum,
    # #                                 aggregate,
    # #                                 per_sample_noise,
    # #                                 respect_attr,
    # #                                 device)
    # # # del model_3
    # # # torch.cuda.empty_cache()

    # mean_cond_1 = mean_cond_1.detach().cpu().numpy()
    # mean_cond_2 = mean_cond_2.detach().cpu().numpy()
    # mean_cond_3 = mean_cond_3.detach().cpu().numpy()
    # sorted_mean_cond_1 = np.sort(mean_cond_1)
    # sorted_mean_cond_2 = np.sort(mean_cond_2)
    # sorted_mean_cond_3 = np.sort(mean_cond_3)
    
    # # PLOTING
    # x_axis = np.arange(1024)
    # fig, ax = plt.subplots()
    # plt.plot(x_axis, sorted_mean_cond_1, label=f"Y-Drop")
    # plt.plot(x_axis, sorted_mean_cond_2, label=f"Drop")
    # plt.plot(x_axis, sorted_mean_cond_3, label=f"Plain")
    # plt.legend(loc='upper left')
    # plt.savefig("mean_conductance_test-set.png")
    # plt.close()







