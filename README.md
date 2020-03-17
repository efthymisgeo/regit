# Add Fancy Title Here

## Repo Structure

```
└── checkpoints
└── configs
└── data
    ├── MNIST/
    ├── CIFAR/
    ├── SVHN/
    └── ImageNet/
└── logs
└── models
    └── regbi.py
└── modules
    ├── CNN
    └── FC
```
- `checkpoints`: contains the checkpoints from the experiments (best model)
- `configs`: has all the necessary config files in order to run the experiment
- `data`: all the datasets that the experiment is carried out
- `logs`: log files from the correspponding experiments which contain the `accuracy`, `accuracies`, `losses`, `switches`, `schedule`, `p_drop`, 
- `models`: the main script which contatins the experiment(s) 
- `modules`: the modules which are used to build the model

