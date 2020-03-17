import os
import six
import json
import torch

MDL_NAMES = [
    "CNN2D",
    "FC"
]


def check_mdl_config(mdl_config):
    """
    Function which checks if mdl_configuration is in valid form
    Args:
        mdl_config (dict)
    """
    for mdl_name, mdl_opts in mdl_config.items():
        if mdl_name not in MDL_NAMES:
            print(f"Found model with name {mdl_name} and parameters {mdl_opts}")
            print(f"Accepting only the names {MDL_NAMES}")
            raise ValueError("Model configuration file contains invalid"
                             "model architecture.")


def load_config(config):
    """Load configuration.
    Args:
        config (dict or .json)
    """
    if isinstance(config, six.string_types):
        with open(config, "r") as f:
            return json.load(f)
    elif isinstance(config, dict):
        return config
    else:
        raise NotImplementedError("Config must be a json file or a dict")


def get_device(exp_config):
    """
    Helper function which gets the device on which the experiment will be run
    Args:
        exp_config(dict)
    """
    try:
        use_cuda = exp_config["use_cuda"]
        cuda_device = exp_config["cuda_device"]
    except KeyError:
        raise NotImplementedError("USE_CUDA flag is missing")

    device = \
        torch.device(
            cuda_device if (use_cuda and torch.cuda.is_available()) else "cpu")
    return device


def get_exp_name(name, exp_config, mdl_config):
    """
    Get Experiment Name
    Args:
        exp_config(dict)
    """
    # TODO
    pass


def print_config(_dict):
    """outputs dict keys and values
    """
    for k, v in _dict.items():
        print(f"{k} \n \t {v}\n")


def get_kwargs(exp_config):
    try:
        use_cuda = exp_config["use_cuda"]
    except KeyError:
        raise NotImplementedError("USE_CUDA flag is missing")
    return {'num_workers': 1, 'pin_memory': False} if use_cuda else {}