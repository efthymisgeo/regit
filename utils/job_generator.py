import os
import sys
import copy
import argparse

def query_yes_no(question, default="yes"):
        """Ask a yes/no question via raw_input() and return their answer.
        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
        The "answer" return value is True for "yes" or False for "no".
        """
        valid = {"yes": True, "y": True, "ye": True,
                "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                "(or 'y' or 'n').\n")

if __name__ == "__main__":
    # ##########################################################
    # # Read arguments
    # ##########################################################
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-j', required=True,
    #                     help="Job name")
    # parser.add_argument('-t', required=True, default="11:59:00",
    #                     help="HH:MM:SS Estimated time the job will take.")
    # parser.add_argument('-e',
    #                     help="conda environment to be activated.")
    # parser.add_argument('-c', required=True,
    #                     help="conda environment to be activated.")
    # parser.add_argument('-p', type=str, default="pascal",
    #                     help="pascal partition has a max time-limit of 36 hours."
    #                         "To get round this, use the pascal-long partition.")
    # parser.add_argument('-g', type=int, default=1,
    #                     help="number of gpus.")
    # parser.add_argument('--high', dest='high', action='store_true',
    #                     help='Use the high priority account "T2-CS055-GPU".'
    #                         'The default is low "T2-CS055-SL4-GPU"')

    # opt = parser.parse_args()

    # if opt.high:
    #     account = "T2-CS055-GPU"
    # else:
    #     account = "T2-CS055-SL4-GPU"

    job_path = "aris"
    job_name = "x-large-mul-na"
    exp_type = "na" 
    peak_list = [(8, 10), (13, 16), (19, 22), (25, 28), (31, 34), (37, 40)]
    sigma_list = [0.005, 0.001] # 0.0005, 0.0001]

    head = f"""
#!/bin/bash"""

    header = f"""
####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name={job_name}    # Job name
#SBATCH --ntasks=2     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=2     # Tasks per node
#SBATCH --cpus-per-task=1     # Threads per task
#SBATCH --time=32:00:00   # walltime
#SBATCH --mem=20G   # memory per NODE
#SBATCH --gres=gpu:2 # GPUs per node to be allocated
#SBATCH --partition=gpu    # Partition
#SBATCH --account=pa181004    # Replace with your system project"""

    pre_body = f"""

export I_MPI_FABRICS=shm:dapl

if [ x$SLURM_CPUS_PER_TASK == x ]; then
export OMP_NUM_THREADS=1
else
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi"""

    body = f"""

## LOAD MODULES ##
module purge            # clean up loaded modules 
  
# load necessary modules
module use ${{HOME}}/modulefiles
module load gnu/8.3.0
module load intel/18.0.5
module load intelmpi/2018.5
module load cuda/10.1.168
module load python/3.6.5
module load pytorch/1.3.1
module load slp/1.3.1"""

    pre_footer = f"""

## RUN YOUR PROGRAM ##
    """

    wait = f"""
wait"""

    runner = head.strip() + header + pre_body + body + pre_footer
    for peak in peak_list:
        for sigma in sigma_list:
            tmp_runner = runner
            command1 = \
                f"""
srun --gres=gpu:1 --mem=10G --ntasks=1 -o logs/${{SLURM_JOB_NAME}}_{peak[0]}_{exp_type}_{sigma}_${{SLURM_JOB_ID}}.out -e logs/${{SLURM_JOB_NAME}}_{peak[0]}_{exp_type}_{sigma}_${{SLURM_JOB_ID}}.err  python models/regbi.py -m configs/model/layers-80sec_deep3.json -d configs/dataset/cifar10.json -e configs/experiment/aris/x-large_mul_{peak[0]}_{exp_type}_{sigma}.json &"""
            command2 = \
                f"""
srun --gres=gpu:1 --mem=10G --ntasks=1 -o logs/${{SLURM_JOB_NAME}}_{peak[1]}_{exp_type}_{sigma}_${{SLURM_JOB_ID}}.out -e logs/${{SLURM_JOB_NAME}}_{peak[1]}_{exp_type}_{sigma}_${{SLURM_JOB_ID}}.err  python models/regbi.py -m configs/model/layers-80sec_deep3.json -d configs/dataset/cifar10.json -e configs/experiment/aris/x-large_mul_{peak[1]}_{exp_type}_{sigma}.json &"""
            tmp_runner = tmp_runner + command1 + command2 + wait
    
            write_approval = True 
            # query_yes_no(f"IS THE GENERATED SCRIPT OK? \n\n" + "=" * 50 +
            #                             f"\n\n\n {tmp_runner}", default="no")
            
            job_conf_name = f"job_x-large_na_{str(sigma)}_{str(peak[0])}_{str(peak[1])}.sh"
            job_conf_path = os.path.join(job_path, job_conf_name)
            if write_approval:
                with open(job_conf_path, "w") as f:
                    f.write(tmp_runner)

                #ex_approval = query_yes_no(f"Execute the job '{opt.j}' ?", default="no")

                #if ex_approval:
                #    os.system(f"sbatch {opt.j}.sh")

            else:
                print("Exiting...")
