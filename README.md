# Add Fancy Title Here

## Prerequisites

### Dependencies
- Python Version >= 3.6
- PyTorch Version >=
- Captum Version >=

### Install Requirements
add a requirements.txt here etc

## Repo Structure

### Bird's Eye View

```
└── checkpoints
└── configs
└── data
    ├── MNIST/
    ├── CIFAR/
    ├── SVHN/
    └── ImageNet/
└── experiments
└── models
    └── regbi.py
└── modules
    ├── CNN
    └── FC
```
- `checkpoints`: contains the checkpoints from the experiments (best model)
- `configs`: has all the necessary config files in order to run the experiment
- `data`: all the datasets that the experiment is carried out
- `experiments`: log files from the correspponding experiments which contain the `accuracy`, `accuracies`, `losses`, `switches`, `schedule`, `p_drop`, 
- `models`: the main script which contatins the experiment(s) 
- `modules`: the modules which are used to build the model

Each of the `checkpoints`, `data` and `experiments` folders have the following
format
```
└── <folder_name>
    ├── MNIST/
    ├── CIFAR/
    ├── SVHN/
    └── ImageNet/
```
where each subfolder contains the corresponding checkopoint models, data and experiments respectivelly.

### Configuration Files

The folder `configs` consists of three subfolders namely
- `dataset`: conf files which specify which dataset will be used  
- `model`: specifies the model architecture
- `experiment`: specifies the kind of experiment that will be carried out 

## Training Models

The command that should be used for training the models is of the following form.
```bash
python models/regbi.py \
-m <path/to/model/conf> \
-d <path/to/data/conf> \
-e <path/to/experiment/conf>
```
e.g

```bash
python models/regbi.py -m configs/model/layers-80sec_shallow.json -d configs/dataset/cifar10.json -e configs/experiment/osc_condrop_cifar10.json
```

