import os
import sys
import json
import torch
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utils.config_loader import *


def load_experiment_options():
    """Helper function which loads all experiment options, namely:
        1. model options
        2. experimental options
        3. dataset options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mdl_config", required=True,
                        help="Path to model configuration file"),
    parser.add_argument("-e", "--exp_config", required=False,
                        default="configs/exp_config.json",
                        help="Path to experiment configuration file."
                             "Handles optimizer, lr, batch size, etc")
    parser.add_argument("-d", "--dt_config", required=False,
                        default="configs/mnist_config.json",
                        help="Path to data cnfiguration file")
    
    args = parser.parse_args()
    mdl_config = load_config(args.mdl_config)
    exp_config = load_config(args.exp_config)
    data_config = load_config(args.dt_config)

    check_mdl_config(mdl_config)    
    exp_config["device"] = get_device(exp_config)
    #exp_config["model_name"] = get_exp_name(exp_config)
    
    exp_config["kwargs"] = get_kwargs(exp_config)

    exp_options = {"model_opt": mdl_config,
                   "exp_opt": exp_config,
                   "data_opt":data_config}

    return exp_options
