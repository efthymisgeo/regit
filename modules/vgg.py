########################################
# Custom modifiaction of VGG source code
########################################
import os
import sys
import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from modules.idrop import ConDropout

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, p_drop=0.5):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=p_drop),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=p_drop),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _set_requires_grad(self, requires_grad=False, module="featrues"):
        """Sets the gradients of a layer/module/model to the argument flag
        Args:
            requires_grad (bool): the flag which handles the gradient pass
            # TODO remove module functionality
            module (str): available choices are "all" and "features"
        """
        for param in self.features.parameters():
            param.requires_grad = requires_grad
        for param in self.classifier.parameters():
            param.requires_grad = requires_grad 

    def _reset_clf(self, model, clf_params, reinit=False, avgpool=True):
        """construct a vgg classifier
        Args:
            model (torch.nn.Module): the module which consists of the
                pre-trained model parameters
            clf_params (dict): a dict which has all the necessary classifier info
            reinit (bool): handles reinitializing opiton. can be false only when
                resuming training in the same dataset (ImageNet)
        """
        # get params for the new model
        params = clf_params["FC"]
        fc_list = params["fc_layers"]
        p_bucket = params["p_bucket"] 
        fc = []
        # load pretrained classifier
        pretrained_classifier = model.classifier
        pre_fc_list = []
        for k in model.classifier:
            if isinstance(k, nn.Linear):
                pre_fc_list.append(k)
        # append linear layers
        for i_fc in range(0, len(fc_list)-2):
            if reinit:
                fc.append(nn.Linear(fc_list[i_fc], fc_list[i_fc+1]))
            else:
                fc.append(pre_fc_list[i_fc])
            fc.append(nn.ReLU(True))
            if p_bucket != 0.0:
                if params["algorithm"] == "dropout":
                    fc.append(nn.Dropout(params["p_bucket"][i_fc]))
                elif params["algorithm"] == "idrop":
                    fc.append(ConDropout(cont_pdf=params["method"],
                                         p_buckets=params["p_bucket"],
                                         n_buckets=params.get("n_buckets", 2),
                                         inv_trick=params["inv_trick"],
                                         alpha=params["alpha"],
                                         drop_low=params["drop_low"],
                                         rk_history=params["rk_history"],
                                         prior=params["prior"],
                                         sigma_drop=params["sigma_drop"]))
                elif params["algorithm"] == "plain":
                    pass
                else:
                    raise NotImplementedError("Not a valid regularization" 
                        "algorithm. You should check your model conf file.")
        if reinit:
            fc.append(nn.Linear(fc_list[-2], fc_list[-1]))
        else:
            fc.append(pre_fc_list[-1])
        self.classifier = nn.Sequential(*fc)
        if reinit:
            self._initialize_weights(modules="classifier")
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) 
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        
    def forward(self, x, rankings=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if rankings is None: 
            x = self.classifier(x)
        else:
            cnt = 0
            for m in self.classifier.children():
                if isinstance(m, ConDropout):
                    x = m(x, rankings[cnt])
                    cnt += 1
                else:
                    x = m(x)
        return x

    def _initialize_weights(self, modules="classifier"):
        """modified initialization function
        Args:
            modules (str): string-id which handles the modules which will be 
                re-initialized. Available choices are "all", "classifier" 
        """
        if modules == "all":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        else:
            # classifier case
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress,
         requires_grad, grad_module, new_arch, **kwargs):
    """Function which constructs a VGG-based architecture
    Args:
        requires_grad (bool): flag which handles the trainable parts of the net
        grad_module (str): id-string which is used to specify which modules
            will be frozen
        new_arch (dict): a dict with all the necessary classifier info
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    #import pdb; pdb.set_trace()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    if not requires_grad:
        model._set_requires_grad(requires_grad=requires_grad,
                                 module=grad_module)
    if new_arch != {}:
        model._reset_clf(model, new_arch, reinit=False)

    return model


def vgg11(pretrained=False, progress=True, requires_grad=False,
          grad_module="features", new_arch={}, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress,
                requires_grad, grad_module, new_arch, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)