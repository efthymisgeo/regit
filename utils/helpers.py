import numpy as np
# load data
from torchvision import datasets
# load the training data

def get_mean_std_of_dataset(data_name=CIFAR10,
                            data_path="data/CIFAR10",
                            max_val=255):
    """
    This function gets mean and std of an image dataset. Note that we will
    further need to use a snippet like the following
    >> mean = [125.3, 123.0, 113.9]
    >> std = [63.0, 62.1, 66.7]
    >> max_val = 255  # images in [0, 255] range (MNIST, CIFAR)
    >> mean_norm = [m/max_val for m in mean]
    >> std_norm = [m/max_val for m in mean]
    """
    train_data = datasets.CIFAR10('./cifar10_data', train=True, download=True)
    # use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    # print(x)
    print(x.shape)
    # calculate the mean and std along the (0, 1) axes
    train_mean = list(np.mean(x, axis=(0, 1)))
    train_std = list(np.std(x, axis=(0, 1), ddof=1))
    # then the mean and std
    mean_norm = [i/max_val for i in train_mean]
    std_norm = [i/max_val for i in train_std]
    print(mean_norm, std_norm)
    return mean_norm, std_norm 