#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --account=def-weimin
#SBATCH --mem=80000M   # 80 GB of RAM
#SBATCH --mem=80000M   # 80 GB of RAM
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12      # CPU cores/threads
# Request a single task, which will utilize the resources
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out   # standard output
#SBATCH --error=%x-%j.err    # standard error
#SBATCH --job-name=infer-zer-shot
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job Started on $(date)"
nvidia-smi

# Load needed python and cuda modules
module load python/3.11 cuda cudnn
module load mpi4py

# Activate your enviroment
source ~/envs/py311/bin/activate

# Run the python script
python ~/projects/inference_sst/inference-zero-shot/inference_zero-shot.py

echo "Job finished on $(date)"