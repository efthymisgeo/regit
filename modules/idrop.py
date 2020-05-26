import os
import sys
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        unit_ranking (torch.tensor): a tensor which has the relative rankings
            of importance for all the neurons in the given layer. Its default
            value is None which drops to the original dropout case for
            debugging purposes.
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
                 unit_ranking=None,
                 p_buckets=[0.25, 0.75],
                 n_buckets=2,
                 cont_pdf=None,
                 inv_trick="dropout"):
        super(ConDropout, self).__init__()
        self.unit_ranking = unit_ranking
        self.p_buckets = p_buckets.sort()
        self.n_buckets = n_buckets
        self.cont_pdf = cont_pdf
        self.inv_trick = inv_trick
        self.p_mean = mean(self.p_buckets)
        
        self.split_intervals = self._get_bucket_intervals()

        for i, p in enumerate(p_buckets):
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
        for i in enumerate(self.split_intervals[:-1]):
            p_masks_low = masks > self.split_intervals[i]
            p_masks_high = masks < self.split_intervals[i+1]
            # input is p_drop but we want p_keep = 1 - p_drop
            prob_masks(p_masks_low & p_masks_high) = 1 - self.p_buckets[i]
        return prob_masks

    def generate_bucket_mask(self, prob_masks):
        n_units = prob_masks.size(1)
        for i in enumerate(self.split_intervals[:-1]):
            start_idx = np.floor(self.split_intervals[i] * n_units)
            end_idx = np.floor(self.split_intervals[i+1] * n_units)
            prob_masks[:,start_idx:end_idx] = 1 - self.p_buckets[i]
        return prob_masks

    def sort_units(self, input):
        """Function which sorts units based on their ranking
        """
        _, sorted_idx = self.unit_ranking.sort()
        return sorted_idx
    
    def get_masks(self, input):
        if self.unit_ranking is None:
            # dropout case: randomly set a neuron ranking
            prob_masks = input.data.new(input.size()).uniform_(0.0, 1.0)
            prob_masks = self.generate_random_masks(prob_masks)
        else:
            sorted_units = self.sort_units(input)
            prob_masks = input.data.new_ones(input.size())
            prob_masks = self.generate_bucket_mask(prob_masks)
            prob_masks = prob_masks[sorted_units]
        # sample Be(p_interval)
        bin_masks = torch.bernoulli(prob_masks)
        # scaling trick
        if self.inv_trick == "dropout":
            masks = torch.div(bin_masks, 1 - self.p_mean)
        else:
            masks = torch.div(bin_masks, prob_masks)
        return masks
            
    def forward(self, input):
        if self.training:
            masks = self.get_masks(input)
            output = torch.mul(input, masks)
        else:
            # at inference the layer is deactivated
            output = input
        return output