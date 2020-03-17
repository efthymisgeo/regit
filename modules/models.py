import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config
from functions import *

NON_LINEARITIES = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'none': None
}

class BinaryLinear(nn.Module):

    #def forward(self, input):
    #    binary_weight = binarize(self.weight)
    #    if self.bias is None:
    #        return F.linear(input, binary_weight)
    #    else:
    #        return F.linear(input, binary_weight, self.bias)
    
    def __init__(self, input_features, output_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.linear = nn.Linear(input_features,
                                output_features,
                                bias=bias)
        self.activ = binarize
    
    def forward(self, input):
        #if self.bias is None:
        #    return binarize(F.linear(input, self.weight))
        #else:
        #    return binarize(F.linear(input, self.weight, self.bias))
        return self.activ(self.linear(input))

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class DNN(nn.Module):

    def __init__(self, input_size, layers, store_activations=False):
        super(DNN, self).__init__()
        self.n_layers = len(layers)
        self.store_activations = store_activations
        fc = [nn.Linear(input_size, layers[0])]
        for i in range(1, self.n_layers):
            fc.append(nn.Linear(layers[i-1], layers[i]))
        self.fc = nn.ModuleList(fc)

    @staticmethod
    def get_activated_nodes(x):
        x_act = x.clone().cpu().detach().numpy()
        return x_act

    def forward(self, x):
        # if the input is from MNIST
        if len(x.shape) == 4:
            x = x.view(-1, 28 * 28)
        if self.store_activations:
            activations = []
            for i in range(self.n_layers - 1):
                x = self.fc[i](x)
                activations.append(self.get_activated_nodes(x))
                x = F.relu(x)
            x = self.fc[-1](x)
            activations.append(self.get_activated_nodes(x))
            x_activations = activations[0]
            for layer_activations in activations[1:]:
                x_activations = np.append(x_activations,
                                          layer_activations, axis=1)
        else:
            x_activations = None
            for i in range(self.n_layers - 1):
                x = F.relu(self.fc[i](x))
            x = self.fc[-1](x)
        return x_activations, F.log_softmax(x, dim=1)


class CNN2D(nn.Module):
    """
    CNN2D model architecture - Convolutional layers followed by fully connected

    Args:
        input_shape (tuple): shape of input image (height, width, channels)
        kernels (list): number of kernels per conv layer
        kernel_size (int/tuple): size of kernels to be applied
        stride (list): list of stride values for every conv layer
        padding (list): list of padding values for every conv layer
        maxpool (bool): handles the use of maxpooling
        pool_size (list): 2D pool size [x-axis, y-axis]
        conv_drop (list): handles the drop probability in 
            convolutional layers
        p_conv_drop (float): drop probability for conv layers
        conv_batch_norm (bool): handles the use of batch norm
        regularization (bool): add or not regularization
        activation (str): defines which non-linearity will be used
        fc_layers (list): list with number of units per layer
        add_dropout (bool): add or ont dropout
        p_drop (float): drop probability for fc layers
        device (str): cpu or gpu usage
    """

    def __init__(self, input_shape=[28, 28], kernels=[40, 100], kernel_size=2, stride=1, padding=0,
                 maxpool=True, pool_size=[2, 2], conv_drop=False,
                 p_conv_drop=0.2, conv_batch_norm=False, regularization=True,
                 activation="relu", fc_layers=[100,100], add_dropout=True,
                 p_drop=0.5, device="cpu"):
        super(CNN2D, self).__init__()
        # load cnn parameters
        self.input_shape = input_shape
        self.kernels = kernels
        self.channels = self._get_conv_channels()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool = maxpool
        self.pool_size = pool_size
        self.conv_drop = conv_drop
        self.p_conv_drop = p_conv_drop
        self.conv_batch_norm = conv_batch_norm
        self.regularization = regularization
        self.activation = activation
        # load fc parameters
        self.fc_layer_list = fc_layers
        self.add_dropout = add_dropout
        self.p_drop = p_drop
        self.n_fc_layers = len(fc_layers)

        # construct modules
        self.device = device
        self.conv, self.conv_out = self._make_conv()
        self.fc_input_size = int(np.prod(self.conv_out)) # channels x height x width
        self.fc = self._make_fc()

        self.drop_masks = []
        
        self.fc_1_idx = {str(i): 0 for i in range(self.fc_layer_list[0])}
    
    def _get_activ(self):
        """Returns activation function 
        """
        activ = NON_LINEARITIES.get(self.activation, nn.ReLU)
        if activ is not None:
            activ = activ()
        return activ

    def _get_conv_channels(self):
        """Returns input and output channels list of lists 
        """
        if len(self.input_shape) == 2:
            self.input_shape.insert(0, 1)
            print("Appending an extra dimension to" 
                  f"the 2D image as {self.input_shape}")
        elif len(self.input_shape) == 3:
            print(f"Input already in shape {self.input_shape}")
        else:
            raise ValueError(f"Input is {self.input_shape} which is not" 
                             "accepted. Reshape it in 2D or 3D.")
        channel_list = []
        in_channels = self.input_shape[0]
        for i_cnn in range(len(self.kernels)):
            in_out_channel_list = [in_channels, self.kernels[i_cnn]]
            in_channels = self.kernels[i_cnn]
            channel_list.append(in_out_channel_list)
        return channel_list

    def _make_conv(self):
        """Generates convolutional layers from given kernels and kernel size
        """
        conv_layers = []
        in_channels, x, y = self.input_shape

        for i_cnn in range(len(self.channels)):
            conv_layers.append(nn.Conv2d(in_channels=self.channels[i_cnn][0],
                                         out_channels=self.channels[i_cnn][1],
                                         kernel_size=self.kernel_size[i_cnn],
                                         stride=self.stride[i_cnn],
                                         padding=self.padding[i_cnn]))
            # calculate conv output tensor dimensions (x-axis & y-axis)
            x = ((x - self.kernel_size[i_cnn] + 2*self.padding[i_cnn]) /
                 self.stride[i_cnn]) + 1
            y = ((y - self.kernel_size[i_cnn] + 2*self.padding[i_cnn]) /
                 self.stride[i_cnn]) + 1

            if self.conv_batch_norm:
                conv_layers.append(nn.BatchNorm2d(self.channels[i_cnn][1]))
            conv_layers.append(self._get_activ())  # append activation function
            if self.maxpool[i_cnn]:
                conv_layers.append(nn.MaxPool2d(kernel_size=self.pool_size))
                x = x // self.pool_size[0]
                y = y // self.pool_size[1]
            if self.conv_drop[i_cnn]:
                conv_layers.append(nn.Dropout(p=self.p_conv_drop))

        output_size = (self.channels[-1][-1] , x, y)
        return nn.Sequential(*conv_layers), output_size
    
    def _make_fc(self):
        """Generate fully connected layers
        """
        fc_list = self.fc_layer_list.copy() # copy to no affect original list
        fc_list.insert(0, self.fc_input_size) # append the cnn output at the beggining
        fc = []
        # append linear layers
        for i_fc in range(1, len(fc_list)):
            fc.append(nn.Linear(fc_list[i_fc-1], fc_list[i_fc]))
        return nn.ModuleList(fc)

    def set_dropout(self, p, mix_rates=False):
        if mix_rates:
            #print(f"added mixed rates --{p[0]}: random-- & --{p[1]}: intel--")
            self.p_drop = p[0] + p[1]
        else:
            self.p_drop = p

    def reset_drop_cnt(self):
        self.fc_1_idx = {str(i): 0 for i in range(self.fc_layer_list[0])} 
    
    def init_mask(self, trick="bernoulli", p_drop=None):
        """
        Initializes drop mask with prob p.
        There are three available methods:
        - uniform: where we sample from a unifrom distribution in (0,1) and 
            then threshold based on the prob p
        - deterministic: where we get a random permutation of all 
            indices in the layer and alwyas hold a percentage p of them
            to be dropped
        - bernoulli: where we create a tensor of the hidden layer size with p
            probs everywhere and then sample a uniwue bernoulli for every 
            neuron
        - ones: added for code compatibility issues. Equivalent to not using
            dropout
        
        Args:
            trick (str): "uniform", "deterministic", "bernoulli"
            p_drop (float): dropout probability. by default is none. use it in
                case you want to overwrite the default config value
        """
        if p_drop is None:
            p_drop = self.p_drop
        
        for i, n_neurons in enumerate(self.fc_layer_list):
            if i == (self.n_fc_layers-1): continue
            if trick == "bernoulli":
                temp_mask = torch.ones(n_neurons, device=self.device)
                mask = torch.mul(temp_mask, p_drop)
                self.drop_masks.append(torch.bernoulli(mask))
            elif trick == "deterministic":
                mask_idx = torch.randperm(n_neurons)
                n_keep_idx = int(p_drop*n_neurons)
                drop_idx, _ = torch.sort(mask_idx[:n_keep_idx])
                temp_mask = torch.ones(n_neurons, device=self.device)
                temp_mask[drop_idx] = 0.0
                self.drop_masks.append(temp_mask)
            elif trick == "uniform":
                mask = torch.FloatTensor(n_neurons).uniform_(0,1).to(self.device)
                self.drop_masks.append(((mask > p_drop).int()).float())
            elif trick == "ones":
                temp_mask = torch.ones(n_neurons, device=self.device)
                self.drop_masks.append(temp_mask)
            else:
                raise ValueError("Not implemented trick")

    def update_mask(self, importance=None, p_drop=None, mix_rates=False):
        """
        Updates mask based on some criterion
        Function which handles different dropout scenarios. Namely
            1. Plain Dropout
            2. Intelligent Dropout
            3. Mixout

        Args:
            importance (list): list of tensors contatining the extracted
                importance per neuron. Empty list corresponds to case 1.
            p_drop (float or tuple of floats): value also the dropout rate 
                in the plain dropout case
            mix_rates (bool): use mixing dropout rates (case 3)
        """
        # check for existing dropout prob
        if p_drop is None:
            p_drop = self.p_drop
        
        # clear mask list
        self.drop_masks = []

        if importance is None:
            # corresponds to dropout case
            self.init_mask(trick="bernoulli", p_drop=p_drop)
        elif mix_rates:
            plain_drop_p = p_drop[0]
            intel_drop_p = p_drop[1]
            if plain_drop_p == 0.0:
                # intel mode only
                #print(f"only intel mode is on")
                self.smooth_importance(importance,
                                       method="mean",
                                       p_drop=intel_drop_p)
            else:
                # both modes active
                #print(f"both modes are active")
                self.mixout(importance, plain_drop_p, intel_drop_p)
        else:
            self.smooth_importance(importance, method="mean", p_drop=p_drop)
                    
    def apply_mask(self, x, mask=None, idx=0):
        """
        function that applies a given mask for a given layer given the
        dropout probability for inverse dropout implementation

        Args:
            x: input tensor
            mask: #TODO remove if no longer needed
            idx (int): identifier for the layer under examination
            p_drop (float): the dropout probability of that layer/stage
        """
        # if self.p_drop == 0.0:
        #     masked_output = torch.mul(x, self.drop_masks[idx])
        # else:
        #######################################################################
        ##### configuration similar to pytorch and tf
        #######################################################################
        masked_output = \
            torch.div(torch.mul(x, self.drop_masks[idx]), 1-self.p_drop)
        #print("applied mask")
        #######################################################################
        #### configuration in which only limiting p_drop value is being used
        #######################################################################
        #masked_output = \
        #    torch.div(torch.mul(x, self.drop_masks[idx]), 1-0.5)
        return masked_output
    
    def mixout(self, importance, plain_drop_p, intel_drop_p):
        plain_drop_list = []
        intel_drop_list = []
        for k in self.fc_layer_list:
            plain_drop_list.append(int(k*plain_drop_p))
            intel_drop_list.append(int(k*intel_drop_p))

        for i, lc in enumerate(importance):
            # method to check if more than one elements are nan
            if torch.sum(torch.isnan(lc)) >= intel_drop_list[i]:
                # random pick
                mask_idx = torch.randperm(self.fc_layer_list[i])
                perc = \
                    int((plain_drop_list[i] + intel_drop_list[i]) * self.fc_layer_list[i])
                mixed_idx, _ = torch.sort(mask_idx[:perc])
                #import pdb; pdb.set_trace()
                print("attribution invalid")
            else:
                # Q: is there any way to avoid ranking/topk???
                # intelligent pick first
                _, sorted_idx = torch.topk(lc, intel_drop_list[i], largest=True)
            
                #import pdb; pdb.set_trace()
                ###############################################################
                ##### experimental code for faster implementation
                ###############################################################
                # get a tensor with all idxs
                all_idxs = torch.arange(*lc.size()).to(self.device)
                # keep only the remaining idxs
                intel_mask = torch.ones(*lc.size()).bool().to(self.device)
                intel_mask[sorted_idx] = False
                leftover_idx = torch.masked_select(all_idxs, intel_mask)
                # randomly pick from the remaining 
                rand_perm = torch.randperm(len(leftover_idx)).to(self.device)
                rand_idx = leftover_idx[rand_perm][:plain_drop_list[i]]
                # use all idxs
                mixed_idx = torch.cat((sorted_idx, rand_idx))
            
            temp_mask = torch.ones(*lc.size()).to(self.device)
            temp_mask[mixed_idx] = 0.0
            ###############################################################
            # intel_idx, _ = torch.sort(sorted_idx)
            # #import pdb; pdb.set_trace()
            # # random pick then
            # leftover_idx = \
            #      torch.tensor([i for i in range(self.fc_layer_list[i]) 
            #      if i not in intel_idx]).to(self.device)
            # #import pdb; pdb.set_trace()
            # # LEN HERE MIGH T BE WRONG
            # rand_idx = torch.randperm(len(leftover_idx)).to(self.device)
            # import pdb; pdb.set_trace()
            # n_keep_idx = plain_drop_list[i]
            # rand_idx, _ = torch.sort(rand_idx[:n_keep_idx])

            # # create mask
            # temp_mask = torch.ones(*lc.size()).to(self.device)
            # # geat all idxs
            # mixed_idx, _ = torch.sort(torch.cat((intel_idx, rand_idx)))
            # temp_mask[mixed_idx] = 0.0
            self.drop_masks.append(temp_mask)

            # for debugging purposes
            if i == 0:
                for k in mixed_idx.cpu().detach().numpy():
                    try:
                        self.fc_1_idx[str(k)] += 1
                    except:
                        print(k)
            
    def smooth_importance(self, importance, method="mean", p_drop=None):
        """
        Function which smooths the importance over neurons in a batch.
        Args:
            importance (list): list of tensors which contain the attribution
            method (str): method which will be used in order 
                to smooth attributions
            p_drop (float): default is None which corresponds to plain dropout
        Returns:
        """
        if p_drop is None:
            p_drop = self.p_drop

        k_list = [int(k*p_drop) for k in self.fc_layer_list[:-1]]
        for i, lc in enumerate(importance):
            # average via mean
            #temp_imp = torch.sum(lc, dim=0)
            # sort top k per importance

            ###################################################################
            ##### check if any of the importances is invalid (nan)
            ###################################################################
            # first method to check is [checks whether an input is nan]
            #if torch.isnan(lc).any():
            #if (lc != lc).any(): hack :)
            # second method is to check is more than one elements are nan
            if torch.sum(torch.isnan(lc)) >= k_list[i]:
                # random pick
                mask_idx = torch.randperm(self.fc_layer_list[i])
                unsorted_idx, _ = torch.sort(mask_idx[:k_list[i]])
                print("attribution invalid")
            else:
                # Q: is there any way to avoid ranking/topk???
                _, unsorted_idx = torch.topk(lc, k_list[i], largest=True)
            
                #if torch.max(sorted_idx) == 1065353216:

                 
                #import pdb; pdb.set_trace()
                # get unsorted idxs
                #unsorted_idx, _ = torch.sort(sorted_idx)
            
            # create mask
            temp_mask = torch.ones(self.fc_layer_list[i], device=self.device)
            temp_mask[unsorted_idx] = 0.0
            #import pdb; pdb.set_trace()            
            # add to drop mask
            self.drop_masks.append(temp_mask)
            # for debugging purposes
            if i == 0:
                for k in unsorted_idx.cpu().detach().numpy():
                    try:
                        self.fc_1_idx[str(k)] += 1
                    except:
                        print(k)
                    
    def get_masks(self, x):
        random_sequence = np.random.choice([0, 1],
                                           size=(x.shape[1],),
                                           p=[1-self.binarization_rate,
                                              self.binarization_rate])
        mask = torch.from_numpy(random_sequence).to(self.device)
        mask_indices = torch.arange(0, mask.size(0))[mask].bool().to(self.device)
        #mask_indices = mask_indices.to(self.device)

        #one = torch.tensor(1, dtype=torch.uint8, device=self.device)
        #zero = torch.tensor(0, dtype=torch.uint8, device=self.device)
        
        # find indices by condition at first and then apply them with the random mask
        top_indices = x > self.bin_threshold
        bottom_indices = x <= self.bin_threshold
        
        on_idx = top_indices & mask_indices
        off_idx = bottom_indices & mask_indices

        return on_idx, off_idx

    def regbi(self, x):
        on_mask, off_mask = self.get_masks(x)
        relu_mask = ~(on_mask + off_mask)
        output = x.clone()
        output = dropbin(output, on_mask, off_mask)
        output[relu_mask] = F.relu(x[relu_mask])
        return output

    def droprelu(self, x):
        on_mask, off_mask = self.get_masks(x)
        relu_mask = ~(on_mask + off_mask)
        output = x.clone()
        output = probrelu(x, on_mask + off_mask)
        output[relu_mask] = F.relu(x[relu_mask])
        return output

    def new_binarize(self, x):
        # normalize x
        #x = (x - x.min()) / (x.max() - x.min())

        # create a mask with ones as defined by binarization rate
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

        # binarize tensor at the specified rate
        x[top_indices & mask_indices] = 1
        x[bottom_indices & mask_indices] = 0
        return x


    def forward(self, x):
        # if the input is not from MNIST
        #import pdb; pdb.set_trace()
        #if x.shape[0] != 64:
        #    print(f"I'm in forward {x.shape}")
        if len(x.shape) != 4:
            x = x.view(x.shape[0], 1, self.input_shape[0], self.input_shape[1])
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # BxCxHxW
        out = self.conv(x)
        # BxD
        out = out.view(-1, self.fc_input_size) 
        
        # apply regularization
        if self.regularization:
            for i in range(self.n_fc_layers -1):
                if self.training:
                    out = F.relu(self.fc[i](out))
                    ###########################################################
                    # usage: mimic dropout
                    ###########################################################
                    out = self.apply_mask(out, idx=i)
                else:
                    ###########################################################
                    #   usage: mimic dropout aka scale
                    ###########################################################
                    #x = torch.mul(self.fc[i](x), self.p_drop)
                    out = self.fc[i](out)
                    out = F.relu(out)
        elif self.add_dropout:
            # original dropout implemetnation
            for i in range(self.n_fc_layers - 1):
                x = self.fc[i](x)
                x = self.drop_list[i](x)
                x = F.relu(x)
        else:
            # inference branch
            for i in range(self.n_fc_layers - 1):
                out = F.relu(self.fc[i](out))
        
        out = self.fc[-1](out)
        return F.log_softmax(out, dim=1)


class CNN1D(nn.Module):

    def __init__(self, input_size, kernels, kernel_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, kernels[0], kernel_size, 1)
        self.conv2 = nn.Conv1d(kernels[0], kernels[1], kernel_size, 1)
        self.fc_input_size = kernels[1]*(((input_size-kernel_size+1)//2-kernel_size+1)//2)
        self.fc1 = nn.Linear(self.fc_input_size, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # batch_size, input_channels, length of signal sequence
        x = x.view(x.shape[0], 1, x.shape[1])
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return None, F.log_softmax(x, dim=1)

