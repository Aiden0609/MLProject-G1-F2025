#!/bin/bash
#SBATCH --time=0-00:20:00
#SBATCH --account=def-weimin
# Request 288000 MB of memory (1/4th of the total node memory)
#SBATCH --mem=288000M
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12      # CPU cores/threads
# Request a single task, which will utilize the resources
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out   # standard output
#SBATCH --error=%x-%j.err    # standard error
#SBATCH --job-name=era-small-aurora
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job Started on $(date)"
nvidia-smi

# Load needed python and cuda modules
#module load python/3.11 cuda cudnn
module load mpi4py

# Activate your enviroment
source ~/envs/py311/bin/activate

# Run the python script
python /home/mridul01/projects/small-aurora/inference-with-metrics/era5-smallAurora.py

echo "Job finished on $(date)"