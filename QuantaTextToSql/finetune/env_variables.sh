# huggingface
export HF_TOKEN="YOUR_HF_TOKEN"

# transformers
export TOKENIZERS_PARALLELISM=false

# accelerate config
export ACCELERATE_DISTRIBUTED_TYPE="MULTI_GPU"
export ACCELERATE_USE_MPS_DEVICE=0
export ACCELERATE_USE_CPU=0
export ACCELERATE_MIXED_PRECISION="yes"
export ACCELERATE_GPU_IDS="all"
export ACCELERATE_MULTI_GPU_BACKEND="gloo"


# wandb disabled
wandb disable