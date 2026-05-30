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

export PATH=/usr/local/bin:$PATH
export https_proxy=http://www-proxy.ijs.si:8080
export http_proxy=http://www-proxy.ijs.si:8080
export no_proxy=127.0.0.0/8,localhost,127.0.0.1,::1
export NO_PROXY=127.0.0.0/8,localhost,127.0.0.1,::1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT=$((8000 + RANDOM % 1000))
echo "Starting vLLM on GPU 0, port $PORT..."

# Run vLLM directly (no srun) — stays alive as a background process on this node
CUDA_VISIBLE_DEVICES=0 singularity exec --nv \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env http_proxy="" \
    --env https_proxy="" \
    --env HTTP_PROXY="" \
    --env HTTPS_PROXY="" \
    --bind /etc/passwd,/etc/group,/dev/shm:/dev/shm out/pytorch_nn.sif \
    vllm serve google/gemma-3-12b-it \
    --enable-lora \
    --max-lora-rank 16 \
    --port $PORT &
VLLM_PID=$!

echo "Waiting for vLLM to initialize..."
while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do
    sleep 10
done
echo "vLLM is ready!"

export VLLM_API_URL="http://localhost:$PORT/v1"
echo "Starting DeepSpeed training on GPUs 1,2,3..."

srun --overlap --ntasks=1 --gpus=3 \
    singularity exec --nv \
    --env CUDA_VISIBLE_DEVICES=1,2,3 \
    --env http_proxy=http://www-proxy.ijs.si:8080 \
    --env https_proxy=http://www-proxy.ijs.si:8080 \
    --env HTTP_PROXY=http://www-proxy.ijs.si:8080 \
    --env HTTPS_PROXY=http://www-proxy.ijs.si:8080 \
    --env no_proxy=127.0.0.0/8,localhost,127.0.0.1,::1 \
    --env NO_PROXY=127.0.0.0/8,localhost,127.0.0.1,::1 \
    --env VLLM_API_URL="http://localhost:$PORT/v1" \
    --bind /etc/passwd,/etc/group,/dev/shm:/dev/shm out/pytorch_nn.sif \
    accelerate launch --config_file accelerate_config.yaml train_deepspeed.py

kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
