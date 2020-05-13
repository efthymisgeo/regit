import torch
import typing
from typing import Any, Callable, List, Tuple, Union
from captum.attr import LayerConductance

class Importance(LayerConductance):
    def __init__(self,
                 forward_func,
                 layer,
                 device_ids=None):
        super(Importance, self).__init__(forward_func, layer, device_ids)
        #self.mem_old = False
        self.mem_list = []

    def attribute_noise(self,
                        inputs,
                        baselines=None,
                        target=None,
                        additional_forward_args=None,
                        n_steps=50,
                        method="riemann_trapezoid",
                        internal_batch_size=None,
                        return_convergence_delta=False,
                        sample_batch=None,
                        sigma_input=None,
                        sigma_attr=None,
                        adapt_to_tensor=False,
                        momentum=None,
                        aggregate=True,
                        per_sample_noise=False
                        ):
        """
        Function which adds noise in the attribution itslef. This might seem
        stupid at first sight but one should consider that we are using this
        attribution method during training and thus we aim to add noise  
        Args:
            imputs():
            baselines():
            target():
            additional_forward_args():
            n_steps():
            method():
            internal_batch_size():
            return_convergence_delta():
            -------- additional arguments ---------
            sample_batch(float, optional): a float in (0,1) which indicates the
                proportion of the batch that is about to be sampled
            noise_input(float, optional): this argument is used to add 
            noise_attr(float, optional): this argument is used when we want
                to add gaussian noise of `noise_attr` std at the the
                final attribution itself
            adapt_to_tensor(bool, optional): adapt std to tensors internal std
            momentum(float, optional): a float in (0,1) which is being used to
                keep track of an exponentially moving average
            aggregate (bool): specifies if the attribution will be made for
                every sample in the batch (False) or for the whole batch (True)
            per_sample_noise (bool): when True a different noise vector is being
                added in every sample to enforce the use of different masks
                in the same batch 
        """
        # sample tesnor for faster and noisier approximation
        if sample_batch is not None:
            #print("entered sampling")
            keep_idx = self.sample_idx(inputs, sample_batch)
            inputs = inputs[keep_idx, :]
            baselines = baselines[keep_idx, :]
            target = target[keep_idx]
        if sigma_input is not None:
            inputs = \
                self.add_noise_tensor(inputs,
                                      sigma_input,
                                      adapt_to_tensor,
                                      per_sample_noise,
                                      inputs.size())
        
        att_new = \
            self.attribute(inputs,
                           baselines=baselines,
                           target=target,
                           additional_forward_args=additional_forward_args,
                           n_steps=n_steps, method=method,
                           internal_batch_size=internal_batch_size,
                           return_convergence_delta=return_convergence_delta)
        
        att_orig_size = att_new.size()
        
        if aggregate:
            # sum over all samples in the batch and extract a single importance
            # vector representation for the whole batch
            att_new = torch.sum(att_new, dim=0)
        
        if sigma_attr is not None:
            #print(f"Mean of tensor before is {torch.mean(att_new)}")
            #print(f"Old size is {att_new.size()}")
            att_new = \
                self.add_noise_tensor(att_new,
                                      sigma_attr,
                                      adapt_to_tensor,
                                      per_sample_noise,
                                      att_orig_size)

            #print(f"new size is {att_new.size()}")
            #import pdb; pdb.set_trace()
            #print(f"Mean of tensor after is {torch.mean(att_new)}")
        
        if momentum is not None:
            #if self.mem_old is False:
            #    self.mem_old = True
            # initialize the history with zeros
            if len(self.mem_list) == 0:
                self.mem_list.append(att_new * 0.0)
                #import pdb; pdb.set_trace()
            else:
                #import pdb; pdb.set_trace()
                self.mem_list[0] = att_new * 0.0
            self.update_momentum(att_new, momentum)
            att_new = self.mem_list[0]
            #print(f"Mean of tensor after momentum is {torch.mean(self.mem_list[0])}")
        #import pdb; pdb.set_trace()
        return att_new

    @staticmethod
    def sample_idx(inputs, sample_batch):
        """
        Gets an input tensor of (batch_size, dimension) and shuffles it along
        the batch (zero) dimension and then keeps only a proportion of them.
        """
        batch_size = inputs.size(0)
        keep = int(batch_size * sample_batch)
        keep_idx = torch.randperm(batch_size)[:keep]
        return keep_idx
    
    @staticmethod
    def add_noise_tensor(tensors, std,
                         adapt_to_tensor=False,
                         per_sample_noise=False,
                         tensor_size=None):
        """
        Function which adds white noise to a tensor of zero mean and std
        """
        if adapt_to_tensor:
            # adapt to tensor's mean value
            #tensor_mean_value = torch.mean(tensors).detach().cpu().numpy()
            # adapt to tensor's std values
            tensor_std_value = torch.std(tensors).detach().cpu().numpy()
            std = std * tensor_std_value
        if per_sample_noise:
            noise = tensors.data.new(tensor_size).normal_(0.0, std)
        else:
            noise = tensors.data.new(tensors.size()).normal_(0.0, std)
        return tensors + noise

    def update_momentum(self, new_update, momentum):
        self.mem_list[0] = \
            momentum * new_update + (1 - momentum) * self.mem_list[0]
        
    def reset_memory(self):
        #self.mem_old = False
        self.mem_list = []