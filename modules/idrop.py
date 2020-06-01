import os
import sys
import copy
import torch
import time
import numpy as np
import collections
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from configs.config import Config
from utils.functions import *


class ConDropout(nn.Module):
    """
    This class is a custom mask implementation of the dropout module.
    It uses Bernoulli masks to inject randomness and also the inverse
    probability trick to avoid rescaling during inference.
    
    Args:
        p_buckets (list/float): the drop probability of every bucket in an
            ascending order. when a single bucket is used p_buckets falls into
            the dropout case. Default: [0.25, 0.75]
        n_buckets (int): the number of buckets that the units will be separated
            degault value is 2. Will be removed in the future. For debugging
            reasons
        cont_pdf (str): Default is None. Otherwise one of the following choices
            are available "linear", "sigma"
    """
    def __init__(self,
                 p_buckets=[0.25, 0.75],
                 n_buckets=2,
                 cont_pdf=None,
                 inv_trick="dropout"):
        super(ConDropout, self).__init__()
        if isinstance(p_buckets, list):
            self.p_buckets = p_buckets
        else:
            self.p_buckets = [p_buckets]
        self.p_mean = np.mean(self.p_buckets)   
        self.n_buckets = n_buckets
        self.cont_pdf = cont_pdf
        self.inv_trick = inv_trick
        # self.inv_trick = "mak"
        
        self.split_intervals = self._get_bucket_intervals()

        for i, p in enumerate(self.p_buckets):
            if 1.0 < p or p < 0.0:
                raise ValueError("probability not in range [0,1]")
        
        self._init_message()

    def _get_bucket_intervals(self):
        """
        Returns a list with the corresponding interval bounds for each
        bucket. e.g n_buckets=2 then [0.0, 0.5, 1.0]
        and n_buckets=3 then [0.0, 0.33, 0.66, 1.0]
        The intervals are meant to separate a uniform distribution U[0,1]
        """
        intervals = []  # lower bound
        for i in range(self.n_buckets):
            intervals.append(i/self.n_buckets)
        intervals.append(1.0)
        return intervals
    
    def _init_message(self):
        print(f"{self.n_buckets} bucket(s) will be used "
              f"with prob {self.p_buckets} respectively.")
        
    def generate_random_masks(self, prob_masks):
        """function which generates mask based on a given prob mask 
        """
        # we need sorted p_bucket list loop over all but last
        old_masks = prob_masks.data.new_ones(prob_masks.size()).bool()
        out_masks = prob_masks.data.new_ones(prob_masks.size())
        for i, _ in enumerate(self.split_intervals[:-1]):
            p_masks_low = (prob_masks > self.split_intervals[i]).reshape(prob_masks.size())
            p_masks_high = (prob_masks < self.split_intervals[i+1]).reshape(prob_masks.size())
            # input is p_drop but we want p_keep = 1 - p_drop
            old_masks = old_masks & (p_masks_low & p_masks_high)
            #print(old_masks)
            out_masks[old_masks] = 1 - self.p_buckets[i]
            old_masks = ~old_masks
        return out_masks

    def generate_bucket_mask(self, prob_masks):
        n_units = prob_masks.size(1)
        for i, _ in enumerate(self.split_intervals[:-1]):
            start_idx = int(np.floor(self.split_intervals[i] * n_units))
            end_idx = int(np.floor(self.split_intervals[i+1] * n_units))
            prob_masks[:,start_idx:end_idx] = 1 - self.p_buckets[i]
        return prob_masks

    def sort_units(self, input, ranking):
        """Function which sorts units based on their ranking and returns the
        shifted sorted indices
        """
        _, sorted_idx = ranking.sort()
        shift_idx = \
            torch.arange(0, input.size(0), device=input.device).view(-1, 1) * input.size(1)
        _, idx_mapping = sorted_idx.sort()
        return (idx_mapping + shift_idx ).view(-1)
    
    def get_masks(self, input, ranking):
        """
        Args:
            input (torch.tensor): 
            ranking (torch.tensor): a tensor which has the relative rankings
                of importance for all the neurons in the given layer.
                Its default value is None which drops to the original dropout
                case for debugging purposes.
        """
        if ranking is None:
            # dropout case: randomly set a neuron ranking
            prob_masks = input.data.new(input.size()).uniform_(0.0, 1.0)
            prob_masks = self.generate_random_masks(prob_masks)
        else:
            # i-drop case
            sorted_units_transform = self.sort_units(input, ranking)
            prob_masks = \
                self.generate_bucket_mask(input.data.new_ones(input.size()))
            # import pdb; pdb.set_trace()
            prob_masks = \
                prob_masks.view(-1)[sorted_units_transform].reshape(input.size(0),
                                                                    input.size(1))
            #print(f"Mixed Probabilistic Masks are {prob_masks}")
            # print(f"The ranking used is {self.unit_ranking}")
            # print(f"the sorted indices are {sorted_units_transform}")
            # print(f"Probabilistic Masks are {prob_masks}")
            # import pdb; pdb.set_trace()
        # sample Be(p_interval)
        #print(f"Probabilistic Mask is {prob_masks}")
        bin_masks = torch.bernoulli(prob_masks)
        #print(f"Bin Mask is {bin_masks}")
        # scaling trick
        if self.inv_trick == "dropout":
            masks = torch.div(bin_masks, 1 - self.p_mean)
        else:
            masks = torch.div(bin_masks, prob_masks)
        return masks
    
    def _count_buckets(self):
        """
        #todo add a debugginf function which will hold a per bucket counter 
        # for every neuron
        """
        pass
    
    def _count_swithces(self):
        """
        #todo add a debugging function which will hold a per switch (drop) 
        # counter for every neuron
        """
        pass
            
    def forward(self, input, ranking=None):
        if self.training:
            masks = self.get_masks(input, ranking)
            output = torch.mul(input, masks)
        else:
            # at inference the layer is deactivated
            # print("infering")
            output = input
        return output

if __name__ == "__main__":
    # p_drop = 0.5
    # batch_size = 4
    # emd_dim = 5
    # output_flag = True
    # pytorch_layer = nn.Dropout(p=p_drop)
    # custom_layer = ConDropout(p_buckets=p_drop,
    #                           n_buckets=1)
    
    # input_tensor = torch.ones(batch_size, emd_dim)
    # pytorch_layer.train()
    # custom_layer.train()
    # pytorch_sum = 0
    # custom_sum = 0
    # for i in range(10):
    #     pytorch_out = pytorch_layer(input_tensor)
    #     custom_out = custom_layer(input_tensor)
        
    #     if output_flag:
    #         print(torch.mean(pytorch_out, dim=0))
    #         print(torch.mean(pytorch_out, dim=1))
            
    #         print(torch.mean(custom_out, dim=0))
    #         print(torch.mean(custom_out, dim=1))
            
    #         print(pytorch_out)
    #         print(custom_out)
    #         #import pdb; pdb.set_trace()
        
    #     pytorch_sum += torch.mean(pytorch_out)
    #     custom_sum += torch.mean(custom_out)
    
    # print(f"The PyTorch's sum is {pytorch_sum} while our custom is {custom_sum}")

    # # timing comparisson
    # N = 10
    # start_torch = time.time()
    # for i in range(N):
    #     pytorch_out = pytorch_layer(input_tensor)
    # end_torch = time.time()
    # time_torch = end_torch - start_torch
    
    # start_custom = time.time()
    # for i in range(N):
    #     pytorch_out = pytorch_layer(input_tensor)
    # end_custom = time.time()
    # time_custom = end_custom - start_custom

    # print(f"We are {time_custom} while pytorch is {time_torch}")

    # ###########################################################################
    # ####    Buck-Drop
    # ###########################################################################
    # import pdb; pdb.set_trace()
    # print("Buck Drop testing")
    # p_drop = [0.25, 0.75]
    # n_buckets = len(p_drop)
    # buck_drop_layer = ConDropout(p_buckets=p_drop,
    #                              n_buckets=n_buckets)
    
    # buck_drop_layer.train()
    # buck_sum = 0
    # for i in range(10):
    #     buck_out = buck_drop_layer(input_tensor)

    #     if output_flag:
    #         print(torch.mean(buck_out, dim=0))
    #         print(torch.mean(buck_out, dim=1))
            
    #         print(torch.mean(buck_out, dim=0))
    #         print(torch.mean(buck_out, dim=1))
            
    #         print(buck_out)
    #         #import pdb; pdb.set_trace()
        
    #     buck_sum += torch.mean(buck_out)
    # print(f"The PyTorch's sum is {pytorch_sum} while our custom is {custom_sum} and bucket is {buck_sum}")
    
    # ###########################################################################
    # ##### idrop implementation
    # ###########################################################################
    # ranking = torch.randint(1, 10, input_tensor.size())
    # p_drop = [0.25, 0.75]
    # n_buckets = len(p_drop)
    # print(ranking)
    # ddrop_layer = ConDropout(unit_ranking=ranking,
    #                          p_buckets=p_drop,
    #                          n_buckets=n_buckets)
    # ddrop_layer.train()
    # ddrop_sum = 0
    # for i in range(10):
    #     ranking = torch.randint(1, 10, input_tensor.size())
    #     ddrop_out = ddrop_layer(input_tensor, ranking)

    #     if output_flag:
    #         print(torch.mean(ddrop_out, dim=0))
    #         print(torch.mean(ddrop_out, dim=1))
            
    #         print(torch.mean(ddrop_out, dim=0))
    #         print(torch.mean(ddrop_out, dim=1))
            
    #         print(ddrop_out)
    #         #import pdb; pdb.set_trace()
        
    #     ddrop_sum += torch.mean(ddrop_out)
    # print(f"The i-bucket drop is {ddrop_sum}")

    # ddrop_layer.eval()
    # ddrop_out = ddrop_layer(input_tensor, ranking)

    ###########################################################################
    #### gradient lookup
    ###########################################################################
    pytorch_affine_model = nn.Sequential(nn.Linear(10, 3),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(3,1))
    custom_affine_model = nn.Sequential(nn.Linear(10, 3),
                                        ConDropout(p_buckets=0.5, n_buckets=1),
                                        nn.Linear(3,1))
    torch.nn.init.ones_(pytorch_affine_model[0].weight)
    torch.nn.init.ones_(custom_affine_model[0].weight)
    torch.nn.init.constant_(pytorch_affine_model[2].weight, 1.5)
    torch.nn.init.constant_(custom_affine_model[2].weight, 1.5)
    print(pytorch_affine_model[0].weight)
    inp1 = torch.rand(10)
    inp2 = inp1.clone().detach()
    pytorch_out = pytorch_affine_model(inp1)
    print(f"Pytorch's output is {pytorch_out}")
    custom_out = custom_affine_model(inp2)
    print(f"Custom output is {custom_out}")
    pytorch_out.backward()
    custom_out.backward()

    weight_pytorch = pytorch_affine_model[0].weight
    weight_pytorch_grad = pytorch_affine_model[0].weight.grad
    weight_custom = custom_affine_model[0].weight
    weight_custom_grad = custom_affine_model[0].weight.grad

    print(f"PyTorch weight is {weight_pytorch} and gradient is {weight_pytorch_grad}")
    print(f"Custom weight is {weight_custom} and gradient is {weight_custom_grad}")

    ###########################################################################
    #### mask gradients
    ###########################################################################
    pasok = torch.rand(1)
    pasok.requires_grad_(True)
    l = torch.randn(1)
    l.requires_grad_(True)
    z = l * pasok * pasok
    mask = torch.zeros(1)
    mask.requires_grad_(True)
    print(z)
    out = z * z * mask
    print(out)
    out.backward()
    print(f"Intermid Gradient is {z.grad} and mask gradient is {mask.grad} while layer grad is {l.grad}")
        