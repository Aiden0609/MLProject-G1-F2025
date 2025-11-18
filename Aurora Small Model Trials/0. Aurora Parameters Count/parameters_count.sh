#!/bin/bash
#SBATCH --time=0-00:55:00
#SBATCH --account=def-weimin
# Request 288000 MB of memory (1/4th of the total node memory)
#SBATCH --mem=288000M
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12      # CPU cores/threads
# Request a single task, which will utilize the resources
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out   # standard output
#SBATCH --error=%x-%j.err    # standard error
#SBATCH --job-name=aurora-params-count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job Started on $(date)"
nvidia-smi

# --- NEW: Set the Hugging Face Cache Location ---
# This line tells huggingface_hub to store all models/datasets in the specified directory.
# This prevents filling up your default home directory quota.
export HF_HOME="/home/mridul01/scratch/cache"
echo "Hugging Face cache set to: $HF_HOME"

# Load needed python and cuda modules
module load python/3.11 cuda cudnn
module load mpi4py

# Activate your enviroment
source ~/envs/py311/bin/activate

# Run the python script
python /home/mridul01/projects/small-aurora/parameters_count/parameters_count.py

echo "Job finished on $(date)"