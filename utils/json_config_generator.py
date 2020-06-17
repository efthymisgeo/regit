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
    json_template_path2 = "configs/experiment"
    json_template_name = "v2_aris_idrop.json"
    temp_json = os.path.join(json_template_path2, json_template_name)
    
    with open(temp_json, "r") as fd:
        conf = json.load(fd)
    
    experiment_name = "bucket_"
    type_of_experiment = "medium"
    bucket_list = [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.05, 0.95), (0.6, 0.4)]
    noise_list = [0.0, 0.001, 0.005, 0.01, 0.1]

    for noise in noise_list:
        cnt = 0
        for _ in bucket_list:
            new_conf = conf.copy()
            ###################################################################
            #### SET GAUSS ATTRIBUTION VALUES
            ###################################################################
            # new_conf["attribution"]["adapt_to_tensor"] = True
            # new_conf["attribution"]["per_sample_noise"] = True
            # new_conf["attribution"]["respect_attr"] = True
            new_conf["idrop"]["method"] = "bucket"
            new_conf["idrop"]["inv_trick"] = "dropout"
            new_conf["idrop"]["p_buckets"] = \
                [bucket_list[cnt][0], bucket_list[cnt][1]]
            # set experiment id
            new_conf["experiment_id"] = \
                experiment_name + str(bucket_list[cnt][0]) + "_" \
                + str(bucket_list[cnt][1]) + "_" + type_of_experiment \
                + "_" + str(noise) + "_" + str(new_conf["runs"]) + "runs"
            # set new .json name
            new_conf_name = experiment_name + "_" + str(bucket_list[cnt][0]) \
                + "_" + str(bucket_list[cnt][1]) + "_" + type_of_experiment \
                + "_" + str(noise) + ".json"
            cnt = cnt + 1
            new_conf_path = os.path.join(json_template_path, new_conf_name)
            with open(new_conf_path, "w") as o_fd:
                json.dump(new_conf, o_fd, indent=4)

