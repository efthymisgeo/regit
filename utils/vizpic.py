import os
import sys
import copy
import json
import argparse
import torchvision
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utils.mnist import MNIST, CIFAR10
from utils.config_loader import load_config, get_kwargs

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()

def add_noise(img, std):
    img = img + np.random.normal(0, std)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_config", required=False,
                        default="configs/exp_config.json",
                        help="Path to experiment configuration file."
                             "Handles optimizer, lr, batch size, etc")
    parser.add_argument("-d", "--dt_config", required=False,
                        default="configs/mnist_config.json",
                        help="Path to data cnfiguration file")
    args = parser.parse_args()
    exp_config = load_config(args.exp_config)
    data_config = load_config(args.dt_config)
    exp_config["kwargs"] = get_kwargs(exp_config)
    
    data = CIFAR10(data_config, exp_config)
    train_loader, val_loader = data.get_train_val_loaders()
    test_loader = data.get_test_loader()

    classes = \
        ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
         'frog', 'horse', 'ship', 'truck')
    
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    fig, ax = plt.subplots()
    # show images
    imshow(torchvision.utils.make_grid(images))
    name = "_cifar.png"
    plt.savefig("clean"+name, bbox_inches='tight')
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    std = [25, 10, 2, 1.5, 1, 0.5, 0.1, 0.01, 0.001]
    for i, s in enumerate(std):
        noised_images = add_noise(images, s)
        imshow(torchvision.utils.make_grid(noised_images))
        plt.savefig("noisy_"+str(s)+name, bbox_inches='tight')
