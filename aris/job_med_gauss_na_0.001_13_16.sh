#!/bin/bash
####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=gauss-med-mul-na    # Job name
#SBATCH --ntasks=2     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=2     # Tasks per node
#SBATCH --cpus-per-task=1     # Threads per task
#SBATCH --time=32:00:00   # walltime
#SBATCH --mem=20G   # memory per NODE
#SBATCH --gres=gpu:2 # GPUs per node to be allocated
#SBATCH --partition=gpu    # Partition
#SBATCH --account=pa181004    # Replace with your system project

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
    
srun --gres=gpu:1 --mem=10G --ntasks=1 -o logs/${SLURM_JOB_NAME}_13_na_0.001_${SLURM_JOB_ID}.out -e logs/${SLURM_JOB_NAME}_13_na_0.001_${SLURM_JOB_ID}.err  python models/regbi.py -m configs/model/layers-80sec_deep.json -d configs/dataset/cifar10.json -e configs/experiment/aris/gauss_mul_13_na_0.001.json &
srun --gres=gpu:1 --mem=10G --ntasks=1 -o logs/${SLURM_JOB_NAME}_16_na_0.001_${SLURM_JOB_ID}.out -e logs/${SLURM_JOB_NAME}_16_na_0.001_${SLURM_JOB_ID}.err  python models/regbi.py -m configs/model/layers-80sec_deep.json -d configs/dataset/cifar10.json -e configs/experiment/aris/gauss_mul_16_na_0.001.json &
wait