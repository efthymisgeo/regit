import os
import math
import torch


class Config:
    def __init__(self):
        self.use_model = 'CNN2D'  # use 'DNN' or 'CNN2D' or 'CNN1D' 
        #######################################################################
        #self.model_id = "_fast_intel_drop_test_2"
        #self.model_id = "_mixout_10runs_25steps_drop_long_intel"
        #self.model_id = "_curriculum_dropout_50_runs_lr_sch_0"
        #######################################################################
        # uncomment following lines in case of intel drop
        #######################################################################
        self.regularization = True
        self.importance = True  # use importance as priors
        self.use_drop_schedule = True  # schedule on dropout probability
        self.mixout = False  # enable/disable mixout mode
        self.plain_dropout_flag = False # run experiment with original dropout
        self.prob_scheduler = "Exp" # "Lin" -- "Exp" -- "Mul" -- "Step"
        self.gamma = 0.0001  # scheduler hyperparameter
        self.runs = 10  # runs to get statistics
        self.setup = "debug_smartdrop" # baptize your experiment
        self.model_id = self.setup + self.prob_scheduler + "_" \
                        + str(self.gamma) + "_" \
                        + str(self.runs) + "runs"
        self.fc_layers = [1000, 1000, 1000, 800, 10]
        self.kernels = [40, 100]  # taken into account only for CNN model
        self.kernel_size = [5, 5]
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.maxpool = (True, True)
        self.pool_size = (2, 2)
        self.conv_drop = (False, False)
        self.conv_batch_norm = False
        self.p_conv_drop = 0.1
        self.activation = "relu"
        self.input_shape = [28, 28]
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 40 # 40
        self.add_dropout = False  # True: if you want to use dropout
        self.p_drop = 0.5
        # self.lr = 0.01 # O.01
        # self.lr_sgd_slow = 0.01 #0.005 
        # self.lr_adam = 1e-4  # "slow" learning mode
        # self.lr_adam_fast = 1e-4  # 1e-3 "fast" learning mode
        self.weight_decay=0.0
        #self.momentum = 0.9  # typical values 0.8-0.99
        #self.momentum_raw = 0.9
        self.optimizer = "SGD"  # ["Adam", "SGD"]
        self._pick_optimizer()
        #self.optimizer_bin = "SGD"
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = True
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:1" if self.use_cuda else "cpu")
        #self.device = torch.device("cpu")
        self.kwargs = {'num_workers': 1, 'pin_memory': False} if self.use_cuda else {}
        self.valid_size = 0.25
        self.patience = 9
        self.patience_adam = 6
        self.shuffle = True  # False to generate activations
        self.aligned = False  # True to generate activation
        self.saved_model_path = self.ROOT_DIR + '/saved_models/mnist_' \
                                + self.use_model + '_' \
                                + str(self.model_id) + '.pt'
    
    def _pick_optimizer(self):
        """Function to modify optimizer and optim parameters
        """
        if self.optimizer == "SGD":
            self.lr =  0.01
            self.momentum = 0.9
        elif self.optimizer == "Adam":
            self.lr = 1e-4