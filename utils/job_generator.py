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
    job_name = "bucket_"
    exp_type = "medium" 
    bucket_list = [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.05, 0.95), (0.6, 0.4)]
    noise_list = [0.0, 0.001, 0.005, 0.01, 0.1]

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
    for noise in noise_list:
        cnt = 0
        for _, buck in enumerate(bucket_list[:-3]):
            buck_1 = bucket_list[cnt]
            buck_2 = bucket_list[cnt + 1]
            cnt = cnt + 2
            tmp_runner = runner
            command1 = \
                f"""
srun --gres=gpu:1 --mem=10G --ntasks=1 -o logs/${{SLURM_JOB_NAME}}_{buck_1[0]}_{buck_1[1]}_{exp_type}_{noise}_${{SLURM_JOB_ID}}.out -e logs/${{SLURM_JOB_NAME}}_{buck_1[0]}_{buck_1[1]}_{exp_type}_{noise}_${{SLURM_JOB_ID}}.err  python models/regul.py -m configs/model/layers-80sec_deep.json -d configs/dataset/cifar10.json -e configs/experiment/aris/{job_name}_{buck_1[0]}_{buck_1[1]}_{exp_type}_{noise}.json &"""
            command2 = \
                f"""
srun --gres=gpu:1 --mem=10G --ntasks=1 -o logs/${{SLURM_JOB_NAME}}_{buck_2[0]}_{buck_2[1]}_{exp_type}_{noise}_${{SLURM_JOB_ID}}.out -e logs/${{SLURM_JOB_NAME}}_{buck_2[0]}_{buck_2[1]}_{exp_type}_{noise}_${{SLURM_JOB_ID}}.err  python models/regul.py -m configs/model/layers-80sec_deep.json -d configs/dataset/cifar10.json -e configs/experiment/aris/{job_name}_{buck_2[0]}_{buck_2[1]}_{exp_type}_{noise}.json &"""
            tmp_runner = tmp_runner + command1 + command2 + wait
    
            write_approval = True 
            # query_yes_no(f"IS THE GENERATED SCRIPT OK? \n\n" + "=" * 50 +
            #                             f"\n\n\n {tmp_runner}", default="no")
            
            job_conf_name = "job_" + f"{str(cnt)}_" + job_name + f"{str(exp_type)}_{str(noise)}.sh"
            job_conf_path = os.path.join(job_path, job_conf_name)
            if write_approval:
                with open(job_conf_path, "w") as f:
                    f.write(tmp_runner)

                #ex_approval = query_yes_no(f"Execute the job '{opt.j}' ?", default="no")

                #if ex_approval:
                #    os.system(f"sbatch {opt.j}.sh")

            else:
                print("Exiting...")
