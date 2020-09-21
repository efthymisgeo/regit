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

SUPPORTED_DISTRIBUTIONS = [
    "sigma-norm",
    "min-max",
    "bucket"
]

class ConDropout(nn.Module):
    """
    This class is a custom mask implementation of the dropout module.
    It uses Bernoulli masks to inject randomness and also the inverse
    probability trick to avoid rescaling during inference.
    
    Args:
        p_buckets(list/float): the drop probability of every bucket in an
            ascending order. when a single bucket is used p_buckets falls into
            the dropout case. Default: [0.25, 0.75]
        n_buckets(int): the number of buckets that the units will be separated
            degault value is 2. Will be removed in the future. For debugging
            reasons
        bucket_size (list): a list which contains the size of the corresponding
            buckets. Functionality used for non symmetric setup.
        cont_pdf(str): Default is None. Otherwise one of the following choices
            are available "sigma-norm"
        correction_factor(float): the factor which is used for mean correction
        tollerance(float): when mean approx is acceptable
        inv_trick(string): which inversion trick to use
        beta (float): parameter which handles the change of the mean 
            probability of each neuron
        scheduling (str): 
        rk_history (str): short/long short is used for getting a short memory
            ranking (per batch) for all units while `long` encodes the fact
            that a neuron ranking is based on previous rankings
        mask_prob (str): the mask probability which will be used as the current
            batch mask. "average": for the prob_avg and "induced" for the
            induced probability based on the current units importance
        prior (float): uniform prior value
    """
    def __init__(self,
                 p_buckets=[0.25, 0.75],
                 n_buckets=2,
                 bucket_size=[0.5, 0.5],
                 cont_pdf=None,
                 p_mean=0.5,
                 correction_factor=0.01,
                 tollerance=0.01,
                 inv_trick="dropout",
                 beta=0.999,
                 scheduling="mean",
                 rk_history="short",
                 mask_prob="average",
                 prior=0.5):
        super(ConDropout, self).__init__()
        self.cont_pdf = cont_pdf
        self.rk_history = rk_history
        self.scheduling = scheduling
        self.bucket_size = bucket_size
        if self.cont_pdf not in SUPPORTED_DISTRIBUTIONS:
                raise NotImplementedError("Not a supported pdf")
        elif self.cont_pdf == "bucket":
            if isinstance(p_buckets, list):
                self.p_buckets = p_buckets
            else:
                self.p_buckets = [p_buckets]
            if len(self.p_buckets) == 2:
                self.p_high = self.p_buckets[0]
                self.p_low = self.p_buckets[1]
                # this part calculates mul factors for each bucket
                # in case of scheduling
                self.f_low = self.p_low / np.mean(self.p_buckets)
                self.f_high = self.p_high / np.mean(self.p_buckets)
            self.p_mean = np.mean(self.p_buckets)   
            self.n_buckets = n_buckets
            self.split_intervals = self._get_bucket_intervals()
            #import pdb; pdb.set_trace()

            for i, p in enumerate(self.p_buckets):
                if 1.0 < p or p < 0.0:
                    raise ValueError("probability not in range [0,1]")
        else:
            self.p_mean = p_mean
            if self.p_mean < 0.0 or self.p_mean > 1.0:
                raise ValueError("probability not in range [0,1]")
            self.cf = correction_factor
            self.tollerance = tollerance
        
        
        self.inv_trick = inv_trick
        if self.inv_trick == "exp-average":
            self.beta = beta
            self.prob_avg = None
            self.ongoing_scheduling = False
            #self.mask_prob = "induced"
    
        self._init_message()
        # hack to work. should be changed in future
        self.prior = prior
        self.p_init = prior  # [1.0, 0.5, .01] initial prior keep probability
    
    def extra_repr(self) -> str:
        return f'p_buckets={self.p_buckets}, inv_trick={self.inv_trick}'
    
    def reset_avg_prob(self, prob_avg):
        """Sets average probability to a given value
        """
        self.prob_avg = prob_avg
    
    def reset_beta(self, beta):
        """Sets beta to a given value
        """
        self.beta = beta

    def prob_step(self, p_drop, update="mean"):
        """Function which changes p_drop
        Args:
            p_drop (float/list):
            update(str):
        """
        #import pdb; pdb.set_trace()
        if update == "mean":
            self.p_mean = p_drop
        elif update == "bucket":
            self.p_buckets = [self.p_mean + p_drop, self.p_mean - p_drop]
        elif update == "flip":
            self.p_buckets = [self.p_high - p_drop, self.p_low + p_drop]
        elif update == "re-init":
            self.p_init = p_drop
        elif update == "step":
            if (self.p_high == self.p_mean + p_drop):
                self.p_buckets = [self.p_mean + p_drop, self.p_mean - p_drop]
            else:
                self.p_buckets = [p_drop, p_drop]
            # print(f"STEP PROBS ARE {self.p_buckets}")
        elif update == "from-zero":
            self.ongoing_scheduling = True
            # this part handles scheduling mean from zero
            self.p_buckets = [self.f_high * p_drop, self.f_low * p_drop]
            self.p_init = 1 - p_drop # refers to keep probability
            if self.p_buckets[0] == self.p_high:
                self.ongoing_scheduling = False
                if self.p_init == self.prior:
                    self.prior += 0.0001
                    print(f"manually changing prior from {self.p_init} to"
                          f" {self.prior}. COMMENT FOR DEBUGGING PUPROSES")
        else:
            raise NotImplementedError("Not a valid probability update")
       
    def _get_bucket_intervals(self):
        """
        Returns a list with the corresponding interval bounds for each
        bucket. e.g n_buckets=2 then [0.0, 0.5, 1.0]
        and n_buckets=3 then [0.0, 0.33, 0.66, 1.0]
        The intervals are meant to separate a uniform distribution U[0,1]
        """
        intervals = []  # lower bound
        if self.bucket_size == [0.5, 0.5]:
            for i in range(self.n_buckets):
                intervals.append(i/self.n_buckets)
            intervals.append(1.0)
        else:
            cum_buck = 0.0
            intervals.append(0.0)
            for len_buck in self.bucket_size[:-1]:
                cum_buck += len_buck
                intervals.append(cum_buck)
            intervals.append(1.0)
        return intervals
    
    def _init_message(self):
        if self.cont_pdf == "bucket":
            print(f"{self.n_buckets} bucket(s) will be used "
                  f"with prob {self.p_buckets} respectively.")
        else:
            print(f"Continuous mapping used is {self.cont_pdf} with mean value"
                  f"{self.p_mean} and cf {self.cf}")

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

    @staticmethod
    def _normalize(input):
        """
        Function which normalizes given tensor.
        If the tensor is (B, N) the normalization is done across dim=1
        If tensor is (N) (aggregated ranks) then a signle mean std value 
        is extracted.
        """
        eps = 1e-6
        
        if len(input.size()) == 2:
            mean = torch.mean(input, dim=1, keepdim=True)
            std = torch.std(input, dim=1, keepdim=True) + eps
            #input.sub_(mean).div_(std)
        else:
            mean = torch.mean(input)
            std = torch.std(input) + eps
            #input.sub_(mean).div_(std)
        output = torch.div(torch.sub(input, mean), std)
        return output
    
    @staticmethod
    def _min_max(input):
        """Applies the (x - min(x)) / (max(x) -min(x)) transformation
        """
        eps = 1e-6
        if len(input.size()) == 2:
            in_min, _ = torch.min(input, dim=1, keepdim=True)
            in_max, _ = torch.max(input, dim=1, keepdim=True)
        else:
            in_min = torch.min(input)
            in_max = torch.max(input)
        # import pdb; pdb.set_trace()
        #input.sub_(in_min).div(in_max - in_min + eps)
        output = \
            torch.div(torch.sub(input, in_min),
                      torch.sub(in_max, in_min + eps))
        return output

    def _fix_pdf(self, unit_probs):
        """Function which fixes an induced pdf based upon a correction factor
        """
        if len(unit_probs.size()) == 2:
            mean_prob = torch.mean(unit_probs, dim=1, keepdim=True)
        else:
            mean_prob = torch.mean(unit_probs)
        #TODO investigate tollerance usage
        #mean_mask = torch.abs(self.p_mean - mean_prob) >= self.tollerance
        cf = torch.div(1 - self.p_mean, mean_prob)
        fixed_probs = torch.clamp(torch.mul(unit_probs, cf) ,0.0, 1.0)
        return fixed_probs
    
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
        elif self.cont_pdf == "bucket":
            # bucket-drop case
            # TODO: implement alternative of sorting per label
            sorted_units_transform = self.sort_units(input, ranking)
            #import pdb; pdb.set_trace()
            prob_masks = \
                self.generate_bucket_mask(input.data.new_ones(input.size()))
            prob_masks = \
                prob_masks.view(-1)[sorted_units_transform].reshape(input.size(0),
                                                                    input.size(1))
        else:
            # i-drop case
            #print(ranking.size())
            if self.cont_pdf == "sigma-norm":
                ranking = self._normalize(ranking)
                # keep probability
                prob_masks = 1 - torch.sigmoid(ranking)
            elif self.cont_pdf == "min-max":
                prob_masks = 1 - self._min_max(ranking)    
            prob_masks = self._fix_pdf(prob_masks)
                
            #print(f"Mixed Probabilistic Masks are {prob_masks}")
            prob_masks = prob_masks.expand_as(input)
            #print(f"Mixed Probabilistic Masks AFTER EXPANSION are {prob_masks}")
            # print(f"The ranking used is {self.unit_ranking}")
            # print(f"the sorted indices are {sorted_units_transform}")
            # print(f"Probabilistic Masks are {prob_masks}")
            # import pdb; pdb.set_trace()
        
        # import pdb; pdb.set_trace()
        # TODO: in the mask-per-label case implement
        # possibly the mean case is much better
        if self.inv_trick == "exp-average":
            if (self.prob_avg is None) or (self.p_init == self.prior) or self.ongoing_scheduling:
                if self.prob_avg is not None:
                    print(f"mean prob is {torch.mean(self.prob_avg)}")
                #self.prob_avg = prob_masks.data.new_ones(torch.mean(prob_masks, dim=0).size()) * self.p_mean
                self.prob_avg = prob_masks.data.new_ones(torch.mean(prob_masks, dim=0).size()) * self.p_init
                self.p_init = .3  # change it to get in this branch only once
            else:
                # print(f"Entered moving average branch")
                self.prob_avg = \
                    self.beta * self.prob_avg + (1 - self.beta) * torch.mean(prob_masks, dim=0)
            
            if self.rk_history == "long":
                #import pdb; pdb.set_trace()
                prob_masks = self.prob_avg.expand_as(prob_masks)
                #import pdb; pdb.set_trace()
        elif self.inv_trick == "dropout":
            self.prob_avg = prob_masks.data.new_ones(torch.mean(prob_masks, dim=0).size()) * self.p_mean
        else:
            raise NotImplementedError("not an implemented method")
        
        # sample Be(p_interval)
        #print(f"Probabilistic Mask is {prob_masks}")
        # import pdb; pdb.set_trace()
        bin_masks = torch.bernoulli(prob_masks)
        #print(f"Bin Mask is {bin_masks}")
        #import pdb; pdb.set_trace()
        # scaling trick
        if self.inv_trick == "dropout":
            masks = torch.div(bin_masks, 1 - self.p_mean)
        elif self.inv_trick == "temporal":
            masks = torch.div(bin_masks, prob_masks)
        elif self.inv_trick == "exp-average":
            masks =  torch.div(bin_masks, self.prob_avg)
        else:
            raise NotImplementedError("Invalid inversion rescaling trick")
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

    def get_unit_pdf(self):
        """Helper function which get the current unit pdf in an unsorted manner
        """
        return self.prob_avg.clone().detach().cpu().numpy()
            
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

    ###########################################################################
    ### cont pdf check
    ###########################################################################
    multi = False
    method = "min-max"
    idrop = ConDropout(cont_pdf=method,
                       p_mean=0.5,
                       correction_factor=0.05)
    # Batch_size x N_Units
    inp1 = torch.ones(4, 10)
    if multi:
        rankings = torch.rand(4, 10)
    else:
        rankings = torch.rand(10)
    
    iout = idrop(inp1, rankings)


