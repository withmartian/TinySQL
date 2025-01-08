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
    "withmartian/cs11_valid"
    "withmartian/cs12_valid"
    #"withmartian/cs13_dataset"
)

######### MODEL HYPERPARAMETERS #########
num_train_epochs=3
batch_size=8
gradient_accumulation_steps=1
warmup_steps=50
max_seq_length=512
weight_decay=0.01

# Define the list of learning rates
learning_rates=(2e-5)

# Insert your Hugging Face API token in the .env file

sanitized_project_name=$(echo "$project_name" | tr '/\\:#?%,' '_')

# Average token count per sample: 78.19177777777777
# Minimum token count per sample: 24
# Maximum token count per sample: 178

# Define model and simplified model name for the experiment
model_names=(
    "roneneldan/TinyStories-Instruct-2Layers-33M"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    #"HuggingFaceTB/SmolLM2-135M-Instruct"
    #"HuggingFaceTB/SmolLM2-360M-Instruct"
)

declare -A simplified_model_names=(
    ["roneneldan/TinyStories-Instruct-2Layers-33M"]="TinyStories-2Layers-33M"
    ["Qwen/Qwen2.5-0.5B-Instruct"]="Qwen2.5-0.5B"
    ["meta-llama/Llama-3.2-1B-Instruct"]="Llama3.2-1B"
    ["HuggingFaceTB/SmolLM2-135M-Instruct"]="SmolLM2-135M"
    ["HuggingFaceTB/SmolLM2-360M-Instruct"]="SmolLM2-360M"
)

experiment_counter=1

# Loop through each model and dataset
for model_name in "${model_names[@]}"; do
    simplified_model_name="${simplified_model_names[$model_name]}"
    echo "Original: $model_name"
    echo "Simplified: $simplified_model_name"

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
            mkdir -p "$model_dir"

            # To disable tokenizers before fork warning
            export TOKENIZERS_PARALLELISM=false
            
            # Run the Python script with the appropriate arguments
            #accelerate launch --num_processes 4 \
            python finetune.py --finetune \
                    --model_name "$model_name" \
                    --learning_rate "$lr" \
                    --warmup_steps "$warmup_steps" \
                    --num_train_epochs "$num_train_epochs" \
                    --batch_size "$batch_size" \
                    --gradient_accumulation_steps "$gradient_accumulation_steps" \
                    --weight_decay "$weight_decay" \
                    --dataset_name "$dataset_name" \
                    --output_dir "$model_dir" \
                    --max_seq_length "$max_seq_length" \
                    --wandb_project "sft_${sanitized_project_name}" \
                    --wandb_entity "$wandb_entity" \
                    --wandb_run_name "$experiment_name" > "$model_dir/${experiment_name}.txt"
            echo "Fine-tuning completed. Logs are saved at $model_dir/${experiment_name}.txt"
            echo "---------------------------------------------"
            echo "Experiment $experiment_name completed. Clearing GPU memory."
            ((sub_experiment_counter++))
        done
        ((experiment_counter++))
    done
done
