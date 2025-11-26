#!/bin/bash
#SBATCH --time=01-00:00:00
#SBATCH --account=def-weimin
#SBATCH --mem=288000M   #288 GB of RAM

#SBATCH --cpus-per-task=12      # CPU cores/threads
# Request a single task, which will utilize the resources
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out   # standard output
#SBATCH --error=%x-%j.err    # standard error
#SBATCH --job-name=finetuning-download
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job Started on $(date)"
#nvidia-smi

# Load needed modules
#module load cuda cudnn
module load mpi4py
module load nco   # for ncrcat


# Activate your enviroment
source ~/envs/py311/bin/activate

# Writing a requirement file
#pip freeze > requirements.txt

# Run the python script
python ~/projects/Download\ Finetuning\ Data/finetune-download.py

echo "Job finished on $(date)"
