"""
Anorthosis Activation Function
@author: efthygeo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class RevReLU(Function):
    """
    Custom implementation of reversed (flipped) ReLU function.
    Subcalsses Function class.

    #### RevReLU = min{0,x} ####
    """

    @staticmethod
    def forward(ctx, input, mask=None):
        """
        In the forward pass we receive an input tensor and an optional mask.
        If the mask is not None then we apply the transformation only to the
        masked input and leave the rest unchanged. The transformation applied
        is min{0,input}.

        Args:
            ctx (content object): can be used to stash information for backward
                pass
            input (torch.tensor): input tensor
            mask (torch.tensor): mask tensor

        Output:
            torch.tensor: which is the output of the layer
        """
        ctx.save_for_backward(input, mask)
        output = input.clone()
        input = input.clamp(max=0.0)
        output[mask] = input[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        cut the gradients of the positive masked values
        """
        input, mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.gt(input, 0) & mask] = 0
        return grad_input, None


probrelu = RevReLU.apply



class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input > 0] = 1
        output[input <= 0] = 0
        return output

    @staticmethod
    def backward(cxt, grad_output):
        #print("custom backward")
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize = BinarizeF.apply


class Inference_DropBin(Function):

    @staticmethod
    def forward(ctx, input, p):
        output = input.clone()
        negative_mask = torch.le(output, 0.0)  # relu
        output = torch.add(torch.mul(output, 1-p), p)
        # output[negative_mask] = -p
        output[negative_mask] = 0

        return output

step_relu = Inference_DropBin.apply

class Inference_ProbReLU(Function):

    @staticmethod
    def forward(ctx, input, p):
        output = input.clone()
        # get negative mask
        negative_mask = torch.le(output, 0.0)
        # clip positive values and mul with prob
        output = torch.mul(output.clamp(max=0.0), p)
        # mul original with 1-p
        input = torch.mul(input, 1-p)
        # replace positive values with the original ones
        output[~negative_mask] = input[~negative_mask]
        return output

infer_prob_relu = Inference_ProbReLU.apply

class DropBin(Function):

    @staticmethod
    def forward(ctx, input, on_mask, off_mask):
        # save for backward pass
        ctx.save_for_backward(input, on_mask + off_mask)
        # flatten input
        output = input.clone()
        output[on_mask] = 1
        output[off_mask] = 0

        # define cuda tensors
        #one = torch.tensor(1, dtype=torch.float, device=input.device)
        #zero = torch.tensor(0, dtype=torch.float, device=input.device)
        #import pdb; pdb.set_trace()
        # apply on mask
        #output = torch.where(on_idx.view(-1) == 1,
        #                     one,
        #                     output)
        # apply off mask
        #output = torch.where(off_idx.view(-1) == 1,
        #                     zero,
        #                     output)
        # reshape ouput tensor
        #output = output.view(input.shape[0], input.shape[1])
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # get input tensor and binarized indices
        input, bin_mask = ctx.saved_tensors
        grad_input = grad_output.clone()

        # get clipping mask over the whole input
        clip_mask = torch.lt(input, -6) + torch.gt(input, 6)

        # clip only binarized grads
        clip_bin_mask = clip_mask & bin_mask

        # clip grad of bin neurons whose activ above one or below 0
        grad_input[clip_bin_mask] = 0 
        
        
        # flatten given grad
        #grad_input = grad_output.clone().view(-1)

        # clip all grads

        # get indices 
        #clip_flag = torch.lt(input[bin_idx], 0) + torch.ge(input[bin_idx], 1)
        #print(clip_flag.size())
        #clip_idx = bin_idx[clip_flag]
        #grad_input = grad_output.clone()
        #grad_input[clip_idx] = 0
        return grad_input, None, None

dropbin = DropBin.apply


class Set_One(Function):

    @staticmethod
    def forward(ctx, input):
        output = input
        output = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Set_Zero(Function):

    @staticmethod
    def forward(ctx, input):
        output = input
        output = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

set2one = Set_One.apply
set2zero = Set_Zero.apply


class BinDropout(nn.Module):
    __constants__ = ['binarization_rate', 'bin_threshold' 'inplace']

    def __init__(self, binarization_rate=0.5, bin_threshold=0.0,
                 inplace=False):
        super(BinDropout, self).__init__()
        if binarization_rate < 0 or binarization_rate > 1:
            raise ValueError("binarization dropout probability has to be"
                             "between 0 and 1, "
                             "but got {}".format(binarization_rate))
        self.binarization_rate = binarization_rate
        self.bin_threshold = bin_threshold
        self.inplace = inplace

    def get_masker(self, input):
        random_sequence = np.random.choice([0, 1],
                                           size=(x.shape[1],),
                                           p=[1-self.binarization_rate,
                                              self.binarization_rate])
        mask = torch.from_numpy(random_sequence).to(self.device)
        mask_indices = torch.arange(0, mask.size(0))[mask].byte().to(self.device)
        #mask_indices = mask_indices.to(self.device)

        # find indices by condition at first and then apply them with the random mask
        top_indices = (x > self.bin_threshold)
        bottom_indices = (x <= self.bin_threshold)
        
        on_mask = top_indices & mask_indices
        off_mask = bottom_indices & mask_indices

        return on_mask, off_mask

    def separate_neurons(self, input):
        """applies different activation function at each neuron"""
        on_mask, off_mask = self.get_masker(input)
        relu_mask = ~(on_mask + off_mask)
        return on_mask, off_mask, relu_mask

    def forward(self, input):
        output = input.clone()
        on_mask, off_mask, relu_mask = self.separate_neurons(input)
        # binarize
        output = dropibn(output, on_mask, off_mask)
        # apply relu
        output[relu_mask] = F.relu(input[relu_mask])
        return output

    def extra_repr(self):
        return ("binarization_rate={},"
                "bin_threshold={},"
                "inplace={}".format(self.binarization_rate,
                                    self.bin_threshold,
                                    self.inplace))