#!/bin/bash

# Check if logged in to Hugging Face
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "You are already logged in to Hugging Face."
else
    echo "You are not logged in. Logging in now..."
    huggingface-cli login
fi

# W&B settings
project_name="sql_interp"
wandb_entity="dhruv-gretel"

# Define the list of datasets to train on
dataset_names=(
    "withmartian/cs1_dataset" 
    "withmartian/cs2_dataset" 
    "withmartian/cs3_dataset"
)

######### MODEL HYPERPARAMETERS #########
num_train_epochs=1
batch_size=8
gradient_accumulation_steps=1
warmup_steps=100
max_seq_length=512
weight_decay=0.01

# Define the list of learning rates
learning_rates=(1e-5)

# Insert your Hugging Face API token here
export HF_TOKEN="YOUR_HF_TOKEN"

# accelerate config
#export ACCELERATE_DISTRIBUTED_TYPE="MULTI_GPU"
#export ACCELERATE_USE_MPS_DEVICE=0
#export ACCELERATE_USE_CPU=0
#export ACCELERATE_MIXED_PRECISION="no"
#export ACCELERATE_GPU_IDS="all"
#export ACCELERATE_MULTI_GPU_BACKEND="gloo"

sanitized_project_name=$(echo "$project_name" | tr '/\\:#?%,' '_')

# Average token count per sample: 78.19177777777777
# Minimum token count per sample: 24
# Maximum token count per sample: 178

# Define model and simplified model name for the experiment
model_names=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "roneneldan/TinyStories-Instruct-2Layers-33M"
)

declare -A simplified_model_names=(
    ["roneneldan/TinyStories-Instruct-2Layers-33M"]="TinyStories-2Layers-33M"
    ["Qwen/Qwen2.5-0.5B-Instruct"]="Qwen2.5-0.5B"
    ["meta-llama/Llama-3.2-1B-Instruct"]="Llama3.2-1B"
)

experiment_counter=1

# Loop through each model and dataset
for model_name in "${model_names[@]}"; do
    #simplified_model_name=$(echo "$model_name" | sed -E 's|.*/||; s/[-.]+/_/g')
    simplified_model_name="${simplified_model_names[$model_name]}"
    echo "Original: $model_name"
    echo "Simplified: $simplified_model_name"
    
    # Sanitize the dataset name for use in W&B project
    sanitized_dataset_name=$(echo "$dataset_name" | tr '/\\:#?%,' '_')

    for dataset_name in "${dataset_names[@]}"; do
        sub_experiment_counter=1
        # Loop through each learning rate and run the script
        for lr in "${learning_rates[@]}"; do
            # Create the experiment name like sft_llama3.1_lr_experiment_2.1, 2.2, etc.
            simplified_dataset_name="${dataset_name#*/}"
            simplified_dataset_name="${simplified_dataset_name%_dataset}"
            experiment_name="sft_${project_name}_${simplified_model_name}_${simplified_dataset_name}_experiment_${experiment_counter}.${sub_experiment_counter}"
            
            # Print current experiment details
            echo "---------------------------------------------"
            echo "Running experiment with model: $model_name and dataset: $simplified_dataset_name and learning rate: $lr"
            echo "Experiment name: $experiment_name"

            # Create output directory
            model_dir="models/${experiment_name}"
            # mkdir -p "$model_dir"

            # # To disable tokenizers before fork warning
            # export TOKENIZERS_PARALLELISM=false
            
            # # Run the Python script with the appropriate arguments
            # #accelerate launch --num_processes 4 \
            # #finetune_accelerate.py \
            # python finetune.py \
            #         --model_name "$model_name" \
            #         --learning_rate "$lr" \
            #         --warmup_steps "$warmup_steps" \
            #         --num_train_epochs "$num_train_epochs" \
            #         --batch_size "$batch_size" \
            #         --gradient_accumulation_steps "$gradient_accumulation_steps" \
            #         --weight_decay "$weight_decay" \
            #         --dataset_name "$dataset_name" \
            #         --output_dir "$model_dir" \
            #         --max_seq_length "$max_seq_length" \
            #         --wandb_project "sft_${sanitized_project_name}" \
            #         --wandb_entity "$wandb_entity" \
            #         --wandb_run_name "$experiment_name" > "$model_dir/sft_${experiment_name}.txt"
            # echo "Fine-tuning completed. Logs are saved at $model_dir/sft_${experiment_name}.txt"
            # echo "---------------------------------------------"
            ((sub_experiment_counter++))
        done
        ((experiment_counter++))
    done
done
