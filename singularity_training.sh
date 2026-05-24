#!/bin/bash
#SBATCH --job-name=hs
#SBATCH --output=job_outputs/train.out
#SBATCH --error=job_outputs/train.err
#SBATCH --time=14-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Number of tasks (1 per job)
#SBATCH --gpus=4                   # Number of GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --partition=e8
#sbatch --nodelist=elixir,kt-gpu4

export PATH=/usr/local/bin:$PATH
export https_proxy=http://www-proxy.ijs.si:8080
export http_proxy=http://www-proxy.ijs.si:8080

# CRITICAL FIX: Added localhost and 127.0.0.1 to no_proxy so vLLM API calls stay local
export no_proxy=127.0.0.0/8,localhost,127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Start vLLM in the background on GPU 0 using your Singularity container
# Using a random port helps prevent conflicts if other jobs run on the same node
PORT=$((8000 + RANDOM % 1000))
echo "Starting vLLM on port $PORT..."

# Upgrade vLLM and Transformers into your local user directory
srun singularity exec --nv out/pytorch_nn.sif python -m pip install --user --upgrade vllm transformers

CUDA_VISIBLE_DEVICES=0 singularity exec --nv out/pytorch_nn.sif vllm serve google/gemma-3-12b-it \
    --enable-lora \
    --max-lora-rank 16 \
    --port $PORT &
VLLM_PID=$!

# 2. Wait for vLLM server to be fully ready
echo "Waiting for vLLM to initialize..."
while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do
    sleep 10
done
echo "vLLM is ready!"

# 3. Export the vLLM URL so your Python script can use it
export VLLM_API_URL="http://localhost:$PORT/v1"

# 4. Run the Accelerate training script on the remaining GPUs (GPUs 1, 2, and 3)
echo "Starting DeepSpeed training..."
CUDA_VISIBLE_DEVICES=1,2,3 srun singularity exec --nv out/pytorch_nn.sif \
    accelerate launch --config_file accelerate_config.yaml train_deepspeed.py

# 5. Clean up vLLM when training finishes
kill $VLLM_PID
