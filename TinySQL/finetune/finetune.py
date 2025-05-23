# Import necessary libraries
import os
import sys
import random
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint
from dotenv import load_dotenv

# ML libraries
import torch
import wandb
from trl import SFTTrainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import pipeline
from transformers import TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

warnings.filterwarnings("ignore")

# TinySQL evaluation function imports
sys.path.append("/mnt/foundation-shared/dhruv_gretel_ai/research/sql/quanta_text_to_sql")
from TinySQL.training_data.generate_datasets import dict_to_batchitem
from TinySQL.training_data.generate_cs1 import evaluate_cs1_prediction
from TinySQL.training_data.generate_cs2 import evaluate_cs2_prediction
from TinySQL.training_data.generate_cs3 import evaluate_cs3_prediction

MODELS_WITH_FLASH_ATTENTION = ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",]

# Arguments for the script
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning, Evaluation, and Interactive Script for SQL Interp")
    # Model and training arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name or path")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--num_train_epochs", type=float, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")

    # Dataset and output directories
    parser.add_argument("--dataset_name", type=str, default="withmartian/cs1_dataset", help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default="models/debug", help="Directory to save the model")
    
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="sql_interp", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="debug", help="W&B run name")
    parser.add_argument("--wandb_entity", type=str, default="dhruv-gretel", help="W&B entity name")
    parser.add_argument("--hf_organization", type=str, default="withmartian", help="W&B organization name")

    # Modes are mutually exclusive, set either one of evaluate, interactive, or finetune
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--evaluate", action="store_true", help="Evaluate the model on the test set")
    mode_group.add_argument("--interactive", action="store_true", help="Run the model in interactive mode")
    mode_group.add_argument("--finetune", action="store_true", help="Fine-tune the model (default behavior)")

    # Device, might not be needed
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    # Debugging mode
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def push_model_to_hf(output_dir, repo_name, organization, commit_message="Add final model and tokenizer files", private=True):
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.push_to_hub(
        repo_id=f"{organization}/{repo_name}",
        commit_message=commit_message,
        private=private,
    )
    tokenizer.push_to_hub(repo_id=f"{organization}/{repo_name}", commit_message=commit_message)
    print(f"Model and tokenizer pushed to Hugging Face Hub at {organization}/{repo_name}")

def interactive_mode(args, alpaca_prompt, model, tokenizer):
    print("Running in interactive mode. Type 'exit' to quit.")

    # Initialize the text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # Adjust parameters as needed
        # For example, to enable streaming, you can set parameters here
    )

    while True:
        try:
            instruction = input("Instruction (type 'exit' to quit): ")
            if instruction.lower() == 'exit':
                break
            context = input("Context: ")
            prompt = alpaca_prompt.format(instruction, context)

            # Generate the response
            response = text_generator(
                prompt,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                early_stopping=True,
            )

            # Extract the generated text
            generated_text = response[0]['generated_text'][len(prompt):].strip()

            print("Response:")
            print(generated_text)
            print("\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

class EvaluationCallback(TrainerCallback):
    def __init__(
        self, args, evaluate_fn, trainer, dataset, model, tokenizer, alpaca_prompt,
        max_seq_length, evaluate_cs_function, dataset_type, batch_size
    ):
        super().__init__()
        self.args = args
        self.evaluate_fn = evaluate_fn
        self.trainer = trainer
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.evaluate_cs_function = evaluate_cs_function
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.alpaca_prompt = alpaca_prompt

    def on_evaluate(self, args, state, control, **kwargs):
        self.evaluate_fn(
            self.args, self.dataset, self.model, self.tokenizer, self.alpaca_prompt,
            self.evaluate_cs_function, self.dataset_type, self.batch_size,
            step=state.global_step
        )

def evaluate(args, dataset, model, tokenizer, alpaca_prompt, evaluate_cs_function, dataset_type="validation", batch_size=1, step=0):
    model.eval()
    total_predictions = 0
    correct_predictions = 0
    evaluation_score_sum = 0.0
    gt_score_sum = 0.0

    # Set the padding side
    tokenizer.padding_side = "left"

    # Loop over the dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {dataset_type} set on step {step}"):
        # Get the batch of samples as a dictionary of lists
        batch_samples_dict = dataset[i:i+batch_size]

        # Get the number of samples in the batch
        num_samples = len(next(iter(batch_samples_dict.values())))

        # Convert to a list of dictionaries
        batch_samples = [
            {key: batch_samples_dict[key][idx] for key in batch_samples_dict}
            for idx in range(num_samples)
        ]

        # Prepare prompts for each sample in the batch
        prompts = [
            alpaca_prompt.format(sample["english_prompt"], sample["create_statement"])
            for sample in batch_samples
        ]

        # Tokenize the prompts, use padding and truncation to handle variable-length inputs
        encoding = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        )

        # Move input_ids and attention_mask to the model's device
        input_ids = encoding['input_ids'].to(model.device)
        attention_mask = encoding['attention_mask'].to(model.device)

        # Generate outputs using model.generate
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            temperature=0.5,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            early_stopping=True,
        )

        # Process each generated output in the batch
        for idx, sample in enumerate(batch_samples):
            # Get the generated tokens for this sample
            generated_id = generated_ids[idx]

            # The prompt length may vary due to padding, so we need to find where the prompt ends
            input_length = input_ids.shape[1]

            # Extract the generated tokens after the prompt
            generated_tokens = generated_id[input_length:]

            # Decode the generated tokens to get the predicted SQL
            predicted_sql = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Prepare the item for evaluation
            item = dict_to_batchitem(sample)

            # Evaluate the predicted SQL
            prediction_score = evaluate_cs_function(item, predicted_sql)
            label_score = evaluate_cs_function(item, sample['sql_statement'])

            # Update counters and sums
            if prediction_score == 1.00:
                correct_predictions += 1
            else:
                # Print the incorrect predictions
                print("--------------------------------------------------")
                print(f"Instruction {i + idx} {dataset_type} set: {sample['english_prompt']}")
                print(f"Context {i + idx} {dataset_type} set: {sample['create_statement']}")
                print(f"Ground Truth SQL {i + idx} {dataset_type} set: {sample['sql_statement']}")
                print(f"Predicted SQL {i + idx} {dataset_type} set: {predicted_sql}")
                print("--------------------------------------------------")
            total_predictions += 1
            evaluation_score_sum += prediction_score
            gt_score_sum += label_score

    # Compute percentages with checks to avoid division by zero
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        evaluation_score = (evaluation_score_sum / total_predictions) * 100
        gt_score = (gt_score_sum / total_predictions) * 100
    else:
        accuracy = evaluation_score = gt_score = 0.0

    print(f"{dataset_type.capitalize()} Prediction Accuracy: {accuracy:.2f}%")
    print(f"{dataset_type.capitalize()} Evaluation Score: {evaluation_score:.2f}%")
    print(f"{dataset_type.capitalize()} Ground Truth Score: {gt_score:.2f}%")

    # Log the scores to W&B with the dataset type
    wandb.log(
        {
            f"{dataset_type.capitalize()} Prediction Accuracy": accuracy,
            f"{dataset_type.capitalize()} Evaluation Score": evaluation_score,
            f"{dataset_type.capitalize()} Ground Truth Score": gt_score,
        }
    )

def main():
    # Parse the arguments
    args = parse_args()

    # Load the environment variables
    load_dotenv()

    # Generate a random 7-digit seed
    if args.seed is None:
        seed = random.randrange(100007, 1000000, 10)
        args.seed = seed
    else:
        seed = args.seed

    # Set the seed for reproducibility
    set_seed(seed)
    print("Random seed for this run is:", seed)
    
    # Initialize wandb if in finetune or evaluate mode
    if args.finetune or args.evaluate:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity)
        wandb.config.update({'seed': seed}, allow_val_change=True)
    
    # Load HF token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("Hugging Face authentication token not found. Please set the HF_TOKEN environment variable.")

    # Decide evaluation function based on the dataset
    if "withmartian/cs1_dataset" in args.dataset_name:
        evaluate_cs_function = evaluate_cs1_prediction
    elif "withmartian/cs2_dataset" in args.dataset_name:
        evaluate_cs_function = evaluate_cs2_prediction
    elif "withmartian/cs3_dataset" in args.dataset_name:
        evaluate_cs_function = evaluate_cs3_prediction
    else:
        raise ValueError("Invalid dataset name, this script is only for SQL-interp datasets!")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model and set padding, flash_attn and eval_batch_size
    # (currently supports only llama, qwen and tinystories)
    if args.model_name in MODELS_WITH_FLASH_ATTENTION:
        # qwen model and llama model with flash attention
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        if args.model_name == "meta-llama/Llama-3.2-1B-Instruct":
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            model.config.pad_token_id = tokenizer.pad_token_id
            #tokenizer.padding_side = "right"
        eval_batch_size = 1024
    elif "HuggingFaceTB/SmolLM2-" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        eval_batch_size = 256
    elif "roneneldan/TinyStories-Instruct-" in args.model_name:
        # tiny-stories model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        eval_batch_size = 256
    elif "bm1" in args.model_name and "cs" in args.model_name:
        # bm model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        eval_batch_size = 256
    else:
        raise ValueError("Model not supported!")

    # Use the alpaca prompt
    alpaca_prompt = """### Instruction: {} ### Context: {} ### Response: """

    if args.interactive:
        # Interactive mode
        interactive_mode(args, alpaca_prompt, model, tokenizer)
        return

    if args.evaluate:
        # Evaluation mode
        # Load test dataset
        test_dataset = load_dataset(args.dataset_name, split="test")
        evaluate(
            args=args,
            dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            alpaca_prompt=alpaca_prompt,
            evaluate_cs_function=evaluate_cs_function,
            dataset_type="test",
            batch_size=eval_batch_size,
            step=0,
        )
        wandb.finish()
        return

    # Fine-tuning mode
    if args.finetune:
        # Load datasets
        train_dataset = load_dataset(args.dataset_name, split="train")
        val_dataset = load_dataset(args.dataset_name, split="validation")
        test_dataset = load_dataset(args.dataset_name, split="test")

        # Preprocess the dataset
        def preprocess_function(examples):
            prompts = [
                alpaca_prompt.format(instruction, context)
                for instruction, context in zip(
                    examples["english_prompt"], examples["create_statement"]
                )
            ]
            responses = [sql + tokenizer.eos_token for sql in examples["sql_statement"]]
            prompt_encodings = tokenizer(prompts, truncation=True, max_length=args.max_seq_length, add_special_tokens=True)
            response_encodings = tokenizer(responses, truncation=True, max_length=args.max_seq_length, add_special_tokens=False)

            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for prompt_ids, response_ids in zip(prompt_encodings["input_ids"], response_encodings["input_ids"]):
                total_length = len(prompt_ids) + len(response_ids)
                if total_length > args.max_seq_length:
                    overflow = total_length - args.max_seq_length
                    prompt_ids = prompt_ids[:-overflow]

                input_ids = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + response_ids
                attention_mask = [1] * len(input_ids)

                padding_length = args.max_seq_length - len(input_ids)

                # Left-sided padding
                input_ids += [tokenizer.pad_token_id] * padding_length
                labels += [-100] * padding_length
                attention_mask += [0] * padding_length

                input_ids_list.append(input_ids)
                labels_list.append(labels)
                attention_mask_list.append(attention_mask)
            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
            }

        # Preprocess the datasets
        train_dataset_processed = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names, desc="Preprocessing train dataset")
        train_dataset_processed.set_format(type='torch')
        val_dataset_processed = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names, desc="Preprocessing validation dataset")
        val_dataset_processed.set_format(type='torch')

        # Calculate the number of training steps, evaluation steps and effective batch size
        num_training_examples = len(train_dataset_processed)
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # num_devices is 1 for now, should be updated for data parallelism
        effective_batch_size = (args.batch_size * args.gradient_accumulation_steps)  # * num_devices
        steps_per_epoch = max(1, num_training_examples // effective_batch_size)
        eval_steps = steps_per_epoch // 8
        save_steps = steps_per_epoch // 2
        print(f"Steps per epoch: {steps_per_epoch}, Eval steps: {eval_steps}")

        # Training arguments
        #TODO: Add lr_scheduler_type
        training_args = TrainingArguments(
            # Output and Logging Parameters
            output_dir=args.output_dir,
            logging_dir=f"{args.output_dir}/logs",
            logging_steps=1,
            save_steps=save_steps,
            report_to="wandb",

            # Training Parameters
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            optim="adamw_torch",
            seed=args.seed,

            # Evaluation Parameters
            eval_strategy="steps",
            eval_steps=eval_steps,
        )

        # Initialize the trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_processed,
            eval_dataset=val_dataset_processed,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )

        # Add evaluation callback to the trainer for validation
        evaluation_callback = EvaluationCallback(
                args=args,
                evaluate_fn=evaluate,
                trainer=trainer,
                dataset=val_dataset,
                model=model,
                tokenizer=tokenizer,
                alpaca_prompt=alpaca_prompt,
                max_seq_length=args.max_seq_length,
                evaluate_cs_function=evaluate_cs_function,
                dataset_type="validation",
                batch_size=eval_batch_size,
        )
        trainer.add_callback(evaluation_callback)

        # Evaluate on the test set before training
        evaluate(
            args=args,
            dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            alpaca_prompt=alpaca_prompt,
            evaluate_cs_function=evaluate_cs_function,
            dataset_type="test",
            batch_size=eval_batch_size,
            step=trainer.state.global_step,
        )

        # Train the model and evaluate on the validation set
        # trainer.evaluate()
        trainer.train()
        trainer.evaluate()

        # Evaluate on the test set after training
        evaluate(
            args=args,
            dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            alpaca_prompt=alpaca_prompt,
            evaluate_cs_function=evaluate_cs_function,
            dataset_type="test",
            batch_size=eval_batch_size,
            step=trainer.state.global_step,
        )

        # Save the model and tokenizer locally and push to Hugging Face Hub
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model and tokenizer saved in {args.output_dir}")
        push_model_to_hf(args.output_dir, args.wandb_run_name, args.hf_organization)
        
        # Finish the W&B run
        wandb.finish()

        # Free up GPU memory
        del model
        del tokenizer
        del trainer

if __name__ == "__main__":
    main()
