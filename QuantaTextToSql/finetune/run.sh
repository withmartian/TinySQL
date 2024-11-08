#!/bin/bash

if huggingface-cli whoami > /dev/null 2>&1; then
    echo "You are already logged in to Hugging Face."
else
    echo "You are not logged in. Logging in now..."
    huggingface-cli login
fi

project_name="sql_interp"
dataset_name="dhruvnathawani/cs1_dataset"
num_train_epochs=1
batch_size=128
gradient_accumulation_steps=8

# Insert your Hugging Face API token here
HF_TOKEN="YOUR_HF_TOKEN"
warmup_steps=20
sanitized_project_name=$(echo "$project_name" | tr '/\\:#?%,' '_')

# Average token count per sample: 78.19177777777777
# Minimum token count per sample: 24
# Maximum token count per sample: 178
max_seq_length=512

# Define model and simplified model name for the experiment
model_names=(
    "meta-llama/Llama-3.2-1B-Instruct"
    #"roneneldan/TinyStories-Instruct-2Layers-33M"
    "Qwen/Qwen2-0.5B-Instruct"
)

# "meta-llama/Llama-3.2-1B-Instruct" - 64
# "qwen2/Qwen2-0.5B-Instruct" - 32

# Define the list of learning rates
learning_rates=(2e-05)
#learning_rates=(
# 1e-8  2e-8  5e-8  1e-7  2e-7  5e-7 
# 1e-6  2e-6  5e-6  1e-5  2e-5  5e-5 
# 1e-4  2e-4  5e-4  1e-3  2e-3  5e-3
#)

experiment_counter=1

for model_name in "${model_names[@]}"; do
    simplified_model_name=$(echo "$model_name" | sed -E 's|.*/||; s/[-.]+/_/g')
    echo "Original: $model_name"
    echo "Simplified: $simplified_model_name"
    
    # Sanitize the dataset name for use in W&B project
    sanitized_dataset_name=$(echo "$dataset_name" | tr '/\\:#?%,' '_')

    # Start the sub-experiment counter for llama3.1 at 2.x
    sub_experiment_counter=1

    # Loop through each learning rate and run the script
    for lr in "${learning_rates[@]}"; do
        # Create the experiment name like sft_llama3.1_lr_experiment_2.1, 2.2, etc.
        experiment_name="${project_name}_sft_${simplified_model_name}_experiment_${experiment_counter}.${sub_experiment_counter}"
        
        # Print current experiment details
        echo "---------------------------------------------"
        echo "Running experiment with model: $model_name and learning rate: $lr"
        echo "Experiment name: $experiment_name"

        # Create output directory
        model_dir="models/${experiment_name}"
        mkdir -p "$model_dir"

        # To disable tokenizers before fork warning
        export TOKENIZERS_PARALLELISM=false
        
        # Run the Python script with the appropriate arguments
        # accelerate launch finetune.py --sft --hf_token YOUR_HF_TOKEN
        #accelerate launch \
        #        --num_processes 4 \
        #inetune.py \
        python3 finetune.py \
                --sft \
                --model_name "$model_name" \
                --learning_rate "$lr" \
                --warmup_steps "$warmup_steps" \
                --num_train_epochs "$num_train_epochs" \
                --batch_size "$batch_size" \
                --gradient_accumulation_steps "$gradient_accumulation_steps" \
                --dataset_name "$dataset_name" \
                --output_dir "$model_dir" \
                --hf_token "$HF_TOKEN" \
                --max_seq_length "$max_seq_length" \
                --wandb_project "sft_${sanitized_project_name}" \
                --wandb_run_name "$experiment_name" > "$model_dir/sft_${experiment_name}.txt"
        echo "Fine-tuning completed. Logs are saved at $model_dir/sft_${experiment_name}.txt"
        echo "---------------------------------------------"
        ((sub_experiment_counter++))
    done
    # Increment the main experiment counter for the next model
    ((experiment_counter++))
done
