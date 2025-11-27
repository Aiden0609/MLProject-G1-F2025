#!/bin/bash
#SBATCH --time=07-00:00:00
#SBATCH --account=def-weimin
#SBATCH --mem=288000M
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12      # CPU cores/threads
# Request a single task, which will utilize the resources
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out   # standard output
#SBATCH --error=%x-%j.err    # standard error
#SBATCH --job-name=finetune-sst
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed modules
module load mpi4py

# Activate your enviroment
source ~/envs/py311/bin/activate

# Variables for readability (Compute Canada scratch usage)
logdir=/scratch/$USER/saved
datadir=/scratch/$USER/data
# datadir=$SLURM_TMPDIR   # Uncomment if you want to use node-local storage

# Start TensorBoard
tensorboard --logdir="${logdir}/lightning_logs" \
            --host 0.0.0.0 \
            --load_fast false &
echo "TensorBoard started"

# Run finetuning script
python /home/$USER/projects/MLP\ Decoder/Finetuning-SST/finetune_auroraLite_sst.py 
    

echo "Job finished on $(date)"