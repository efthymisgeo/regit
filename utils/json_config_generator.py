import os
import sys
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))

if __name__ == "__main__":
    #cwd = os.getcwd()
    json_template_path = "configs/experiment/aris"
    json_template_path2 = "configs/experiment/aris2"
    json_template_name = "mul_19_na_0.1.json"
    temp_json = os.path.join(json_template_path, json_template_name)
    
    with open(temp_json, "r") as fd:
        conf = json.load(fd)
    
    experiment_name = "_x-large_mul_"
    type_of_experiment = "na"
    peak_list = [8, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40]
    sigma_list = [0.005, 0.001]

    for peak in peak_list:
        for sigma in sigma_list:
            new_conf = conf.copy()
            ###################################################################
            #### SET GAUSS ATTRIBUTION VALUES
            ###################################################################
            # new_conf["attribution"]["adapt_to_tensor"] = True
            # new_conf["attribution"]["per_sample_noise"] = True
            # new_conf["attribution"]["respect_attr"] = True

            # fix peak epoch
            new_conf["use_drop_schedule"]["peak_epoch"] = peak
            # fix noise type
            new_conf["attribution"]["sigma_attr"] = sigma
            # set experiment id
            new_conf["experiment_id"] = \
                experiment_name + str(peak) + "_" + type_of_experiment \
                + "_" + str(sigma) + "_" + str(new_conf["runs"]) + "runs"
            # set new .json name
            new_conf_name = "x-large_mul_" + str(peak) + "_" \
                            + type_of_experiment + "_" + str(sigma) + ".json"
            new_conf_path = os.path.join(json_template_path, new_conf_name)
            with open(new_conf_path, "w") as o_fd:
                json.dump(new_conf, o_fd, indent=4)

