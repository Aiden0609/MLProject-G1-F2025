#!/bin/bash
#SBATCH --time=0-00:15:00
#SBATCH --account=def-lev
# Request 288000 MB of memory (1/4th of the total node memory)
#SBATCH --mem=288000M
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12      # CPU cores/threads
# Request a single task, which will utilize the resources
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out   # standard output
#SBATCH --error=%x-%j.err    # standard error
#SBATCH --job-name=era-cc-job
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed python and cuda modules
module load python/3.11 cuda cudnn
module load mpi4py

# Activate your enviroment
source ~/ENVS/py311/bin/activate

# Variables for readability
#logdir=/home/bmhod/scratch/saved
#datadir=/home/bmhod/scratch/data
#datadir=$SLURM_TMPDIR

#tensorboard --logdir=${logdir}/lightning_logs --host 0.0.0.0 --load_fast false & \
#    python ~/workspace/pl_mnist_example/train.py \
#    --model Conv \
#    --dataloader MNIST \
#    --batch_size 32 \
#    --epoch 10 \
#    --num_workers 10 \
#    --logdir ${logdir} \
#    --data_dir  ${datadir}
python ~/projects/def-lev/bmhod/training/finetuning_base.py

echo "Job finished on $(date)"