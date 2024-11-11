# This script is used to fine-tune a model with SQL-interp datasets using the Accelerate library

# Import necessary libraries
import sys
import os
import torch
import random
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ignore warnings
warnings.filterwarnings("ignore")

# Pytorch imports
import wandb
from trl import SFTTrainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import TrainerCallback, DefaultDataCollator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# QuantaTextToSql evaluation function imports
sys.path.append("/mnt/foundation-shared/dhruv_gretel_ai/research/sql/quanta_text_to_sql")
from QuantaTextToSql.training_data.generate_cs1 import evaluate_cs1_prediction
from QuantaTextToSql.training_data.generate_cs2 import evaluate_cs2_prediction
from QuantaTextToSql.training_data.generate_cs3 import evaluate_cs3_prediction
from QuantaTextToSql.training_data.generate_datasets import dict_to_batchitem, batchitem_to_dict

# Accelerate imports
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize Accelerator
accelerator = Accelerator()

# List of models that use Flash Attention
MODELS_WITH_FLASH_ATTENTION = ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and/or evaluate a model with custom arguments")

    # Add arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name or path")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--num_train_epochs", type=float, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--dataset_name", type=str, default="withmartian/cs1_dataset", help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default="models/debug", help="Directory to save the model")
    parser.add_argument("--seed", type=int, default=420, help="Random seed for reproducibility")
    parser.add_argument("--wandb_project", type=str, default="sft_sql_interp", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="debug", help="W&B run name")
    parser.add_argument("--wandb_entity", type=str, default="dhruv-gretel", help="W&B entity name")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for fine-tuning")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps for fine-tuning")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # Add flags for fine-tuning and evaluation
    parser.add_argument("--sft", action="store_true", help="Fine-tune the model with SFT")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EvaluationCallback(TrainerCallback):
    def __init__(
        self, args, evaluate_fn, trainer, dataset, model, tokenizer,
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

    def on_evaluate(self, args, state, control, **kwargs):
        eval_dataset = self.dataset
        self.evaluate_fn(
            self.args, eval_dataset, self.model, self.tokenizer, self.max_seq_length,
            self.evaluate_cs_function, self.dataset_type, self.batch_size,
            accelerator=self.trainer.accelerator, step=state.global_step
        )

# Evaluation function
def evaluate(
    args, dataset, model, tokenizer, max_seq_length, evaluate_cs_function,
    dataset_type="validation", batch_size=1024, accelerator=None, step=0
):
    model.eval()
    import ipdb; ipdb.set_trace()
    total_prediction_score = torch.tensor(0.0, device=model.device)
    total_gt_score = torch.tensor(0.0, device=model.device)
    num_samples = torch.tensor(0, dtype=torch.int64, device=model.device)
    correct_predictions = torch.tensor(0, dtype=torch.int64, device=model.device)
    total_predictions = torch.tensor(0, dtype=torch.int64, device=model.device)
    wrong_predictions = []
    right_predictions = []

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        prompt_text = f"### Instruction:\n{example['english_prompt']}\n### Context:\n{example['create_statement']}\n### Response:\n"
        #TODO: truncation=True,
        tokenized = tokenizer(prompt_text, padding="max_length", truncation=True, max_length=max_seq_length)
        tokenized["command_set"] = example["command_set"]
        tokenized["table_name"] = example["table_name"]
        tokenized["create_statement"] = example["create_statement"]
        tokenized["english_prompt"] = example["english_prompt"]
        tokenized["sql_statement"] = example["sql_statement"]
        tokenized["table_fields"] = example["table_fields"]
        tokenized["select"] = example["select"]
        tokenized["order_by"] = example["order_by"]
        return tokenized

    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(tokenize_fn, load_from_cache_file=True, desc="Tokenizing in evaluate()")

    accelerator.wait_for_everyone()

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)

    def custom_collate_fn(batch):
        batch_input_ids = torch.stack([example["input_ids"] for example in batch])
        batch_attention_mask = torch.stack([example["attention_mask"] for example in batch])
        batch_command_set = [example["command_set"] for example in batch]
        batch_table_name = [example["table_name"] for example in batch]
        batch_create_statement = [example["create_statement"] for example in batch]
        batch_english_prompt = [example["english_prompt"] for example in batch]
        batch_sql_statement = [example["sql_statement"] for example in batch]
        batch_table_fields = [example["table_fields"] for example in batch]
        batch_select = [example["select"] for example in batch]
        batch_order_by = [example["order_by"] for example in batch]
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "command_set": batch_command_set,
            "table_name": batch_table_name,
            "create_statement": batch_create_statement,
            "english_prompt": batch_english_prompt,
            "sql_statement": batch_sql_statement,
            "table_fields": batch_table_fields,
            "select": batch_select,
            "order_by": batch_order_by,
        }

    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    ) if accelerator else None

    test_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )

    if accelerator:
        model = accelerator.prepare(model)

    gen_model = model.module if hasattr(model, 'module') else model

    for batch in tqdm(test_dataloader, desc=f"Process ID:{accelerator.process_index} is evaluating with {len(test_dataloader)} batches."):
        inputs = {key: val for key, val in batch.items() if key in ["input_ids", "attention_mask"]}

        with torch.no_grad():
            outputs = accelerator.unwrap_model(gen_model).generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_new_tokens=500,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.5,
                top_p=0.7,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_length:]
        predicted_sqls = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        import ipdb; ipdb.set_trace()

        batch_prediction_scores = []
        batch_gt_scores = []
        for i in range(len(predicted_sqls)):
            predicted_sql = predicted_sqls[i].strip()
            item_batch = {key: value[i] if isinstance(value, (list, torch.Tensor)) else value for key, value in batch.items() if key not in ['input_ids', 'attention_mask']}
            item = dict_to_batchitem(item_batch)
            prediction_score = evaluate_cs_function(item, predicted_sql)
            gt_score = evaluate_cs_function(item, item.sql_statement)
            batch_prediction_scores.append(prediction_score)
            batch_gt_scores.append(gt_score)

            if prediction_score >= 0.99:
                correct_predictions += 1
            total_predictions += 1
            import ipdb; ipdb.set_trace()

            data = {
                "table_name": item.table_name,
                "table_fields": item.table_fields,
                "english_prompt": item.english_prompt,
                "create_statement": item.create_statement,
                "sql_statement": item.sql_statement,
                "predicted_sql": predicted_sql,
                "prediction_score": prediction_score,
                "gt_score": gt_score,
                "command_set": item.command_set,
                "select": item.select,
                "order_by": item.order_by,
            }

            if prediction_score < 1 or prediction_score < gt_score:
                wrong_predictions.append(data)
            else:
                right_predictions.append(data)

        batch_prediction_scores_tensor = torch.tensor(batch_prediction_scores, device=model.device)
        batch_gt_scores_tensor = torch.tensor(batch_gt_scores, device=model.device)

        total_prediction_score += batch_prediction_scores_tensor.sum()
        total_gt_score += batch_gt_scores_tensor.sum()
        num_samples += len(batch_prediction_scores)

    total_prediction_score = accelerator.reduce(total_prediction_score, reduction="sum")
    total_gt_score = accelerator.reduce(total_gt_score, reduction="sum")
    num_samples = accelerator.reduce(num_samples, reduction="sum")
    correct_predictions = accelerator.reduce(correct_predictions, reduction="sum")
    total_predictions = accelerator.reduce(total_predictions, reduction="sum")

    gt_evaluation_score = (total_gt_score.item() / num_samples.item()) * 100 if num_samples.item() > 0 else 0
    prediction_evaluation_score = (total_prediction_score.item() / num_samples.item()) * 100 if num_samples.item() > 0 else 0
    simple_accuracy = (correct_predictions.item() / len(dataset)) * 100 if len(dataset) > 0 else 0

    if dist.is_available() and dist.is_initialized():
        all_wrong_predictions = [None for _ in range(dist.get_world_size())]
        all_right_predictions = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_wrong_predictions, wrong_predictions)
        dist.all_gather_object(all_right_predictions, right_predictions)
    else:
        all_wrong_predictions = [wrong_predictions]
        all_right_predictions = [right_predictions]

    if accelerator.is_main_process:
        flat_wrong_predictions = [item for sublist in all_wrong_predictions for item in sublist]
        flat_right_predictions = [item for sublist in all_right_predictions for item in sublist]
        wrong_predictions_df = pd.DataFrame(flat_wrong_predictions)
        right_predictions_df = pd.DataFrame(flat_right_predictions)

        os.makedirs(f"{args.output_dir}/predictions", exist_ok=True)

        wrong_predictions_df.to_csv(f"{args.output_dir}/predictions/{dataset_type}_wrong_predictions_step_{step}.csv", index=False)
        right_predictions_df.to_csv(f"{args.output_dir}/predictions/{dataset_type}_right_predictions_step_{step}.csv", index=False)

        wandb.log(
            {
                f"{dataset_type.capitalize()} Ground Truth Evaluation Score": gt_evaluation_score,
                f"{dataset_type.capitalize()} Prediction Evaluation Score": prediction_evaluation_score,
                f"{dataset_type.capitalize()} Simple Accuracy": simple_accuracy,
            }
        )

        print(f"{dataset_type.capitalize()} Ground Truth Evaluation Score: {gt_evaluation_score}")
        print(f"{dataset_type.capitalize()} Simple Accuracy: {simple_accuracy}")

    model.train()

    return gt_evaluation_score, prediction_evaluation_score, simple_accuracy

def push_model_to_hf(output_dir, repo_name, organization, commit_message="Add final model and tokenizer files", private=True):
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.push_to_hub(
        repo_id=f"{organization}/{repo_name}", 
        commit_message=commit_message, 
        private=private
    )
    tokenizer.push_to_hub(
        repo_id=f"{organization}/{repo_name}", 
        commit_message=commit_message
    )
    print(f"Model and tokenizer pushed to Hugging Face Hub at {organization}/{repo_name}")

def sft(args):
    # Generate a random 7-digit seed if not provided
    if args.seed is None:
        seed = random.randrange(1000001, 10000000, 2)
        args.seed = seed
    else:
        seed = args.seed

    # Set the seed for reproducibility
    set_seed(seed)

    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name, 
            entity=args.wandb_entity
        )
        wandb.config.update({'seed': seed}, allow_val_change=True)

    hf_token = os.environ.get("HF_TOKEN")

    if hf_token is None:
        raise ValueError("Hugging Face authentication token not found. Please set the HF_TOKEN environment variable.")
    
    try:
        # Decide evaluation function based on the dataset
        if "withmartian/cs1_dataset" in args.dataset_name:
            evaluate_cs_function = evaluate_cs1_prediction
        elif "withmartian/cs2_dataset" in args.dataset_name:
            evaluate_cs_function = evaluate_cs2_prediction
        elif "withmartian/cs3_dataset" in args.dataset_name:
            evaluate_cs_function = evaluate_cs3_prediction
        else:
            raise ValueError("Invalid dataset name, this script is only for training with SQL-interp datasets!")

        train_dataset = load_dataset(args.dataset_name, split="train")
        val_dataset = load_dataset(args.dataset_name, split="validation")

        ############ DEBUGGING ##############
        train_dataset = train_dataset.select(range(10000))
        val_dataset = val_dataset.select(range(100))
        #val_dataset = train_dataset.select(range(100))

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=hf_token)
        tokenizer.model_max_length = args.max_seq_length
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16, 
            use_auth_token=hf_token,
        )
        if args.model_name in MODELS_WITH_FLASH_ATTENTION:
            model.config.attn_implementation = "flash_attention_2"
            eval_batch_size = 32
        else:
            eval_batch_size = 256

        alpaca_prompt = """### Instruction:\n{}\n### Context:\n{}\n### Response:\n"""

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

        train_dataset_processed = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=True,
            desc="Preprocessing train dataset",
        )
        val_dataset_processed = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            load_from_cache_file=True,
            desc="Preprocessing validation dataset",
        )
        train_dataset_processed.set_format(type='torch')
        val_dataset_processed.set_format(type='torch')

        num_training_examples = len(train_dataset_processed)
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps * num_devices
        steps_per_epoch = max(1, num_training_examples // effective_batch_size)
        eval_steps = steps_per_epoch // 8
        save_steps = steps_per_epoch // 2

        if accelerator.is_main_process:
            print(f"Steps per epoch: {steps_per_epoch}, Eval steps: {eval_steps}")

        training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            evaluation_strategy="steps",
            logging_steps=1,
            eval_steps=eval_steps,
            save_steps=save_steps,
            optim="adamw_torch",
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="wandb",
            remove_unused_columns=False,
            bf16=True,
            fp16=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset_processed,
            eval_dataset=val_dataset_processed,
            #data_collator=None,
            max_seq_length=args.max_seq_length,
            args=training_args,
        )

        evaluation_callback = EvaluationCallback(
            args=args,
            evaluate_fn=evaluate,
            trainer=trainer,
            dataset=val_dataset,
            model=model,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            evaluate_cs_function=evaluate_cs_function,
            dataset_type="validation",
            batch_size=eval_batch_size,
        )
        trainer.add_callback(evaluation_callback)

        if accelerator.is_main_process:
            print("Random seed:", seed)

        trainer.evaluate()
        trainer.train()

        if trainer.is_world_process_zero():
            unwrapped_model = model
            unwrapped_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Model and tokenizer saved in {args.output_dir}")
            push_model_to_hf(args.output_dir, args.wandb_run_name, "dhruvnathawani")

    finally:
        if accelerator.is_main_process:
            wandb.finish()

if __name__ == "__main__":
    args = parse_args()

    if args.sft or args.evaluate:
        sft(args)

    if not args.sft and not args.evaluate:
        print("Please specify either --sft, --evaluate, or both")
        sys.exit(1)
