import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import torch.optim as optim
from torch.utils.data import Sampler
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utils.model_utils import train, test, validate, EarlyStopping


# TODO add generic ImageLoader
class ImageLoader():
    def __init__(self, data_setup, exp_setup):
        self.dataset = data_setup
        self.exp_setup = exp_setup
    
    def get_train_loader(self):
        pass

    def get_test_loader(self):
        pass

    def get_train_val_loader(self):
        pass

    def get_transform(self):
        pass

    def shuffle_idxs(self):
        pass


class MNIST:
    def __init__(self, data_setup, exp_setup):
        self.seed = data_setup["seed"]
        self.batch_size = exp_setup["batch_size"]
        self.test_batch_size = exp_setup["test_batch_size"]
        self.val_size = exp_setup["valid_size"]
        self.kwargs = exp_setup["kwargs"]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        
        self.data_dir = \
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ,"data/" + 'MNIST')
        
        self.train_loader, self.val_loader = self.get_train_val_loaders()

        self.test_loader = self.get_test_loader()

    def get_train_loader(self):
        # no validations set split
        train_loader = \
            torch.utils.data.DataLoader(
                datasets.MNIST(self.data_dir, train=True, download=True,
                               transform=self.transform), 
                batch_size=self.batch_size,
                **self.kwargs)
        return train_loader

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                    datasets.MNIST(self.data_dir,
                                   train=False,
                                   transform=self.transform),
                    batch_size=self.test_batch_size, shuffle=False,
                    **self.kwargs) 

    def get_train_val_loaders(self):
        # TODO fix this
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((self.val_size >= 0) and (self.val_size <= 1)), error_msg
        
        # load the dataset
        train_dataset = datasets.MNIST(root=self.data_dir,
                                       train=True,
                                       download=True,
                                       transform=self.transform)

        valid_dataset = datasets.MNIST(root=self.data_dir,
                                       train=True,
                                       download=True,
                                       transform=self.transform)

        # random split
        # TODO add function
        n_train_samples = len(train_dataset)
        indices = list(range(n_train_samples))
        split = int(np.floor(self.val_size * n_train_samples))
        split = n_train_samples - split
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=train_sampler,
                                                   **self.kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=valid_sampler,
                                                   **self.kwargs)

        return train_loader, valid_loader


class CIFAR10:
    def __init__(self, data_setup, exp_setup):
        self.seed = data_setup["seed"]
        self.batch_size = exp_setup["batch_size"]
        self.test_batch_size = exp_setup["test_batch_size"]
        self.val_size = exp_setup["valid_size"]
        self.kwargs = exp_setup["kwargs"]
        self.norm_mean = data_setup["normalization"][0]
        self.norm_std = data_setup["normalization"][1]
        self.augmentation = data_setup.get("augmentation", False)
        
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean,
                                 std=self.norm_std),
            ])
        
        if self.augmentation:
            self.train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.norm_mean,
                                        std=self.norm_std),
                    ])
        else:
            self.train_transform = self.val_transform
        
        self.data_dir = \
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ,"data/" + 'CIFAR10')
        
        self.train_loader, self.val_loader = self.get_train_val_loaders()

        self.test_loader = self.get_test_loader()

    def get_train_loader(self):
        # no validations set split
        train_loader = \
            torch.utils.data.DataLoader(
                datasets.CIFAR10(self.data_dir, train=True, download=True,
                                 transform=self.train_transform), 
                batch_size=self.batch_size,
                **self.kwargs)
        return train_loader

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                    datasets.CIFAR10(self.data_dir,
                                     train=False,
                                     transform=self.val_transform),
                    batch_size=self.test_batch_size, shuffle=False,
                    **self.kwargs) 

    def get_train_val_loaders(self):
        # TODO fix this
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((self.val_size >= 0) and (self.val_size <= 1)), error_msg
        
        # load the dataset
        train_dataset = datasets.CIFAR10(root=self.data_dir,
                                         train=True,
                                         download=True,
                                         transform=self.train_transform)

        valid_dataset = datasets.CIFAR10(root=self.data_dir,
                                         train=True,
                                         download=True,
                                         transform=self.val_transform)

        # random split
        # TODO add function
        n_train_samples = len(train_dataset)
        indices = list(range(n_train_samples))
        split = int(np.floor(self.val_size * n_train_samples))
        split = n_train_samples - split
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=train_sampler,
                                                   **self.kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=valid_sampler,
                                                   **self.kwargs)

        return train_loader, valid_loader




class CIFAR100:
    def __init__(self, data_setup, exp_setup):
        self.seed = data_setup["seed"]
        self.batch_size = exp_setup["batch_size"]
        self.test_batch_size = exp_setup["test_batch_size"]
        self.val_size = exp_setup["valid_size"]
        self.kwargs = exp_setup["kwargs"]
        self.norm_mean = data_setup["normalization"][0]
        self.norm_std = data_setup["normalization"][1]
        self.augmentation = data_setup.get("augmentation", False)
        
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean,
                                 std=self.norm_std),
            ])
        
        if self.augmentation:
            self.train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.norm_mean,
                                        std=self.norm_std),
                    ])
        else:
            self.train_transform = self.val_transform
        
        self.data_dir = \
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ,"data/" + 'CIFAR100')
        
        self.train_loader, self.val_loader = self.get_train_val_loaders()

        self.test_loader = self.get_test_loader()

    def get_train_loader(self):
        # no validations set split
        train_loader = \
            torch.utils.data.DataLoader(
                datasets.CIFAR100(self.data_dir, train=True, download=True,
                                 transform=self.train_transform), 
                batch_size=self.batch_size,
                **self.kwargs)
        return train_loader

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                    datasets.CIFAR100(self.data_dir,
                                     train=False,
                                     transform=self.val_transform),
                    batch_size=self.test_batch_size, shuffle=False,
                    **self.kwargs) 

    def get_train_val_loaders(self):
        # TODO fix this
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((self.val_size >= 0) and (self.val_size <= 1)), error_msg
        
        # load the dataset
        train_dataset = datasets.CIFAR100(root=self.data_dir,
                                         train=True,
                                         download=True,
                                         transform=self.train_transform)

        valid_dataset = datasets.CIFAR100(root=self.data_dir,
                                         train=True,
                                         download=True,
                                         transform=self.val_transform)

        # random split
        # TODO add function
        n_train_samples = len(train_dataset)
        indices = list(range(n_train_samples))
        split = int(np.floor(self.val_size * n_train_samples))
        split = n_train_samples - split
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=train_sampler,
                                                   **self.kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=valid_sampler,
                                                   **self.kwargs)

        return train_loader, valid_loader


class STL10:
    def __init__(self, data_setup, exp_setup):
        self.seed = data_setup["seed"]
        self.batch_size = exp_setup["batch_size"]
        self.test_batch_size = exp_setup["test_batch_size"]
        self.val_size = exp_setup["valid_size"]
        self.kwargs = exp_setup["kwargs"]
        self.norm_mean = data_setup["normalization"][0]
        self.norm_std = data_setup["normalization"][1]
        self.augmentation = data_setup.get("augmentation", False)
        
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean,
                                 std=self.norm_std),
            ])
        
        if self.augmentation:
            self.train_transform = transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.norm_mean,
                                         std=self.norm_std),
                ])
        else:
            self.train_transform = self.val_transform
        
        self.data_dir = \
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ,"data/" + 'STL10')
        
        self.train_loader, self.val_loader = self.get_train_val_loaders()

        self.test_loader = self.get_test_loader()

    def get_train_loader(self):
        # no validations set split
        train_loader = \
            torch.utils.data.DataLoader(
                datasets.STL10(self.data_dir, split="train", download=True,
                                 transform=self.train_transform), 
                batch_size=self.batch_size,
                **self.kwargs)
        return train_loader

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                    datasets.STL10(self.data_dir,
                                   split="test",
                                   transform=self.val_transform),
                    batch_size=self.test_batch_size, shuffle=False,
                    **self.kwargs) 

    def get_train_val_loaders(self):
        # TODO fix this
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((self.val_size >= 0) and (self.val_size <= 1)), error_msg
        
        # load the dataset
        train_dataset = datasets.STL10(root=self.data_dir,
                                       split="train",
                                       download=True,
                                       transform=self.train_transform)

        valid_dataset = datasets.STL10(root=self.data_dir,
                                       split="train",
                                       download=True,
                                       transform=self.val_transform)

        # random split
        # TODO add function
        n_train_samples = len(train_dataset)
        indices = list(range(n_train_samples))
        split = int(np.floor(self.val_size * n_train_samples))
        split = n_train_samples - split
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=train_sampler,
                                                   **self.kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=valid_sampler,
                                                   **self.kwargs)

        return train_loader, valid_loader


class SVHN:
    def __init__(self, data_setup, exp_setup):
        self.seed = data_setup["seed"]
        self.batch_size = exp_setup["batch_size"]
        self.test_batch_size = exp_setup["test_batch_size"]
        self.val_size = exp_setup["valid_size"]
        self.kwargs = exp_setup["kwargs"]
        # self.norm_mean = data_setup["normalization"][0]
        # self.norm_std = data_setup["normalization"][1]
        self.augmentation =  False
        
        
        self.val_transform = transforms.Compose([
            transforms.Scale(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
            ])
            # transforms.Normalize(mean=self.norm_mean,
            #                      std=self.norm_std),
            # ])
        
        self.train_transform = self.val_transform
        
        self.data_dir = \
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ,"data/" + 'SVHN')
        
        self.train_loader, self.val_loader = self.get_train_val_loaders()

        self.test_loader = self.get_test_loader()

    def get_train_loader(self):
        # no validations set split
        train_loader = \
            torch.utils.data.DataLoader(
                datasets.SVHN(self.data_dir,
                              split="train",
                              download=True,
                              transform=self.train_transform), 
                batch_size=self.batch_size,
                **self.kwargs)
        return train_loader

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                    datasets.SVHN(self.data_dir,
                                  split="test",
                                  download=True,
                                  transform=self.val_transform),
                    batch_size=self.test_batch_size, shuffle=False,
                    **self.kwargs) 

    def get_train_val_loaders(self):
        # TODO fix this
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((self.val_size >= 0) and (self.val_size <= 1)), error_msg
        
        # load the dataset
        train_dataset = datasets.SVHN(root=self.data_dir,
                                       split="train",
                                       download=True,
                                       transform=self.train_transform)

        valid_dataset = datasets.SVHN(root=self.data_dir,
                                       split="train",
                                       download=True,
                                       transform=self.val_transform)

        # random split
        # TODO add function
        n_train_samples = len(train_dataset)
        indices = list(range(n_train_samples))
        split = int(np.floor(self.val_size * n_train_samples))
        split = n_train_samples - split
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=train_sampler,
                                                   **self.kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=valid_sampler,
                                                   **self.kwargs)

        return train_loader, valid_loader


class IMAGE_NET:
    def __init__(self, data_setup, exp_setup):
        self.seed = data_setup["seed"]
        self.batch_size = exp_setup["batch_size"]
        self.test_batch_size = exp_setup["test_batch_size"]
        self.val_size = exp_setup["valid_size"]
        self.kwargs = exp_setup["kwargs"]
        self.norm_mean = data_setup["normalization"][0]
        self.norm_std = data_setup["normalization"][1]
        self.augmentation = data_setup.get("augmentation", False)
        
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean,
                                 std=self.norm_std),
            ])
        
        if self.augmentation:
            self.train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.norm_mean,
                                         std=self.norm_std),
                    ])
        else:
            self.train_transform = self.val_transform
        
        self.data_dir = \
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ,"/data/scratch/" + 'efthygeo')
        
        self.train_loader, self.val_loader = self.get_train_val_loaders()

        self.test_loader = self.get_test_loader()

    def get_train_loader(self):
        # no validations set split
        train_loader = \
            torch.utils.data.DataLoader(
                datasets.ImageNet(self.data_dir, split="train",
                                 transform=self.train_transform), 
                batch_size=self.batch_size,
                **self.kwargs)
        return train_loader

    def get_test_loader(self):
        return torch.utils.data.DataLoader(
                    datasets.ImageNet(self.data_dir,
                                      split="val",
                                      transform=self.val_transform),
                    batch_size=self.test_batch_size, shuffle=False,
                    **self.kwargs) 

    def get_train_val_loaders(self):
        # TODO fix this
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((self.val_size >= 0) and (self.val_size <= 1)), error_msg
        
        # import pdb; pdb.set_trace()
        # load the dataset
        train_dataset = datasets.ImageNet(root=self.data_dir,
                                          split="train",
                                          download=False,
                                          transform=self.train_transform)

        valid_dataset = datasets.ImageNet(root=self.data_dir,
                                          split="train",
                                          download=False,
                                          transform=self.val_transform)

        # random split
        # TODO add function
        n_train_samples = len(train_dataset)
        indices = list(range(n_train_samples))
        split = int(np.floor(self.val_size * n_train_samples))
        split = n_train_samples - split
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=train_sampler,
                                                   **self.kwargs,
                                                   drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=valid_sampler,
                                                   **self.kwargs)

        return train_loader, valid_loader
    

class MySequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        rng = len(self.data_source)
        offset = self.data_source[0]
        return iter(range(offset, offset + rng))

    def __len__(self):
        return len(self.data_source)
