#!/bin/bash
#SBATCH --job-name=hs
#SBATCH --output=job_outputs/train.out
#SBATCH --error=job_outputs/train.err
#SBATCH --time=14-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Number of tasks (1 per job)
#SBATCH --gpus=4                  # Number of GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --partition=e8
#sbatch --nodelist=elixir,kt-gpu4


export PATH=/usr/local/bin:$PATH
export https_proxy=http://www-proxy.ijs.si:8080
export http_proxy=http://www-proxy.ijs.si:8080
export no_proxy=127.0.0.0/8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun singularity exec --nv out/pytorch_nn.sif accelerate launch --config_file accelerate_config.yaml train_deepspeed.py
