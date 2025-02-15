#!/bin/bash

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=mul-s-4-na-0.001    # DO NOT FORGET TO CHANGE THIS
#SBATCH --output=logs/mul-s-4-na-0.001.%j.out # DO NOT FORGET TO CHANGE THIS. the job stdout will be dumped here. (%j expands to jobId).
#SBATCH --error=logs/mul-s-4-na-0.001.%j.err # DO NOT FORGET TO CHANGE THIS. the job stdout will be dumped here. (%j expands to jobId).
#SBATCH --ntasks=1     # How many times the command will run. Leave this to 1 unless you know what you are doing
#SBATCH --nodes=1     # The task will break in so many nodes. Use this if you need many GPUs
#SBATCH --gres=gpu:1 # GPUs per node to be allocated
#SBATCH --ntasks-per-node=1     # Same as ntasks
#SBATCH --cpus-per-task=4     # If you need multithreading
#SBATCH --time=24:00:00   # HH:MM:SS Estimated time the job will take. It will be killed if it exceeds the time limit
#SBATCH --mem=8G   # memory to be allocated per NODE
#SBATCH --partition=gpu    # gpu: Job will run on one or more of the nodes in gpu partition. ml: job will run on the ml node
#SBATCH --account=pa181004    # DO NOT CHANGE THIS

export I_MPI_FABRICS=shm:dapl

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi


## LOAD MODULES ##
module purge            # clean up loaded modules 

# load necessary modules
module use ${HOME}/modulefiles
module load gnu/8.3.0
module load intel/18.0.5
module load intelmpi/2018.5
module load cuda/10.1.168
module load python/3.6.5
module load pytorch/1.3.1
module load slp/1.3.1

## RUN YOUR PROGRAM ##
cd .
srun python models/regbi.py \
-m configs/model/layers-80sec_shallow.json \
-d configs/dataset/cifar10.json \
-e configs/experiment/aris/mul_4_na_0.001.json