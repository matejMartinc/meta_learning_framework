#!/bin/bash
#SBATCH --job-name=hs
#SBATCH --output=job_outputs/train.out
#SBATCH --error=job_outputs/train.err
#SBATCH --time=14-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=e8
#SBATCH --nodelist=elixir,kt-gpu4

# 1. Create the output directories if they don't exist
mkdir -p job_outputs
mkdir -p checkpoints_sft

# 2. Set Hugging Face environment variables
# IMPORTANT: Put your actual Hugging Face token here if Gemma 3 is a gated model
export HF_TOKEN="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"

# Set cache dir so you don't re-download the 12B model every job run.
# (Change this path if you have a specific shared scratch space)
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p $HF_HOME

echo "Starting training job on $SLURM_NODELIST"

# 3. Run Singularity
# - We bind $PWD:$PWD and set --pwd to execute in the current directory.
# - We pass the HF_TOKEN environment variable into the container.
srun singularity exec --nv \
    --bind /etc/passwd,/etc/group,/dev/shm:/dev/shm,$PWD:$PWD,$HF_HOME:$HF_HOME \
    --pwd $PWD \
    --env HF_TOKEN=$HF_TOKEN \
    out/pytorch_nn.sif \
    python train_sft_baseline.py