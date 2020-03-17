import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model_utils import train, test, validate, EarlyStopping
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler


class MNIST:

    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = self.config.ROOT_DIR + '/data'

        self.train_loader, self.val_loader = self.get_train_val_loaders()

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.data_dir, train=False, transform=self.transform),
            batch_size=self.config.test_batch_size, shuffle=False,
            **self.config.kwargs)

    def get_train_loader(self):
        # no validations set split
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform),
            batch_size=self.config.batch_size, **self.config.kwargs)
        return train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_val_loaders(self):
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((self.config.valid_size >= 0) and (self.config.valid_size <= 1)), error_msg
        # load the dataset
        train_dataset = datasets.MNIST(root=self.data_dir, train=True,
                                       download=True, transform=self.transform)

        valid_dataset = datasets.MNIST(root=self.data_dir, train=True,
                                       download=True, transform=self.transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.config.valid_size * num_train))
        split = num_train - split
        if self.config.aligned:
            train_idx, valid_idx = indices[:split], indices[split:]
            train_sampler = MySequentialSampler(train_idx)
            valid_sampler = MySequentialSampler(valid_idx)
        else:
            if self.config.shuffle:
                np.random.seed(self.config.seed)
                np.random.shuffle(indices)
            train_idx, valid_idx = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.config.batch_size,
                                                   sampler=train_sampler,
                                                   **self.config.kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.config.batch_size,
                                                   sampler=valid_sampler,
                                                   **self.config.kwargs)

        return train_loader, valid_loader

    def run_training(self):
        # define model
        model = ModelFactory(self.config.use_model, self.config.layers).get_model()
        optimizer = optim.SGD(model.parameters(), lr=self.config.lr, momentum=self.config.momentum)

        # early stopping
        earlystop = EarlyStopping(patience=self.config.patience,
                                  verbose=False,
                                  config=self.config,
                                  model_id=self.config.model_id)
        # training
        for epoch in range(1, self.config.epochs + 1):
            print("Epoch: [{}/{}]".format(epoch, self.config.epochs))
            train(self.config, model, self.train_loader, optimizer, epoch)
            val_loss, _ = validate(self.config, model, self.val_loader)
            earlystop(val_loss, model)
            if earlystop.early_stop:
                print("Early Stopping Training")
                break
            if epoch % 5 == 0:
                test(self.config, model, self.test_loader)
        print("finished training")
        print("Saved model's Performance")
        saved_model = ModelFactory(self.config.use_model, self.config.layers).get_model()
        saved_model.load_state_dict(torch.load(self.config.saved_model_path, map_location=self.config.device))
        test(self.config, saved_model, self.test_loader)

    def generate_act_data_by_dataloader(self, model, data_loader):
        activations_data = []
        if data_loader.dataset.train:
            target_file = self.config.acts_train_file
        else:
            target_file = self.config.acts_test_file
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                activations, output = model(data)

                target_ = target.cpu().detach().numpy()
                target_ = target_.reshape(-1, 1)
                activations_targets = np.append(activations, target_, axis=1)
                activations_data.append(activations_targets)

        data = activations_data[0]
        for activations in activations_data[1:]:
            data = np.append(data, activations, axis=0)

        np.savetxt(target_file, data, delimiter=",")
        print('Activations data generated and stored at: ' + target_file)

    def generate_activations_data(self):
        if not os.path.exists(self.config.activations_data_dir):
            os.makedirs(self.config.activations_data_dir)

        model = ModelFactory(self.config.use_model, self.config.layers, store_activations=True).get_model()
        model.load_state_dict(torch.load(self.config.ROOT_DIR +
                                         '/saved_models/mnist_' + self.config.use_model +
                                         '_' + str(self.config.n_acts) + str(self.config.model_id) + '.pt',
                                         map_location=self.config.device))

        self.generate_act_data_by_dataloader(model, self.train_loader)
        self.generate_act_data_by_dataloader(model, self.test_loader)


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
