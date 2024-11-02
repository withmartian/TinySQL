import sys
import os
import argparse
import torch
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys

sys.path.append(
    "/mnt/foundation-shared/dhruv_gretel_ai/research/sql/quanta_text_to_sql"
)
from training_data.generate_cs1 import evaluate_cs1_prediction
from training_data.generate_cs2 import evaluate_cs2_prediction
from training_data.generate_cs3 import evaluate_cs3_prediction

import wandb
from trl import SFTTrainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from training_data.generate_datasets import dict_to_batchitem, batchitem_to_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune and/or evaluate a model with custom arguments"
    )

    # Add arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=1, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="withmartian/cs1_dataset",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/debug",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--hf_token", type=str, required=True, help="Hugging Face authentication token"
    )
    parser.add_argument(
        "--seed", type=int, default=420, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="sql_interp", help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="debug", help="W&B run name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="dhruv-gretel", help="W&B entity name"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay for fine-tuning"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="Warmup steps for fine-tuning"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # Add flags for fine-tuning and evaluation
    parser.add_argument(
        "--sft", action="store_true", help="Fine-tune the model with SFT"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")

    return parser.parse_args()


class EvaluationCallback(TrainerCallback):
    def __init__(
        self,
        evaluate_fn,
        trainer,
        dataset,
        model,
        tokenizer,
        max_seq_length,
        evaluate_cs_function,
        dataset_type,
    ):
        super().__init__()
        self.evaluate_fn = evaluate_fn
        self.trainer = trainer
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.evaluate_cs_function = evaluate_cs_function
        self.dataset_type = dataset_type

    def on_evaluate(self, args, state, control, **kwargs):
        # During callback evaluation, only evaluate on 25% of the dataset
        eval_dataset = self.dataset.shuffle(seed=42).select(range(int(len(self.dataset) * 0.25)))
        gt_score, prediction_score = self.evaluate_fn(
            args,
            eval_dataset,
            self.model,
            self.tokenizer,
            self.max_seq_length,
            self.evaluate_cs_function,
            self.dataset_type,
        )


def evaluate(
    args,
    dataset,
    model,
    tokenizer,
    max_seq_length,
    evaluate_cs_function,
    dataset_type="validation",
    batch_size=1024,
):
    model.eval()
    total_prediction_score = 0
    total_gt_score = 0
    num_samples = 0

    # Set tokenizer padding side and special tokens
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        prompt_text = f"### Instruction:\n{example['english_prompt']}\n### Context:\n{example['create_statement']}\n### Response:\n"
        tokenized = tokenizer(
            prompt_text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        tokenized["command_set"] = example["command_set"]
        tokenized["table_name"] = example["table_name"]
        tokenized["create_statement"] = example["create_statement"]
        tokenized["english_prompt"] = example["english_prompt"]
        tokenized["sql_statement"] = example["sql_statement"]
        tokenized["table_fields"] = example["table_fields"]
        tokenized["select"] = example["select"]
        tokenized["order_by"] = example["order_by"]
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn)
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True
    )

    def custom_collate_fn(batch):
        # Stack input_ids and attention_mask tensors
        batch_input_ids = torch.stack([example["input_ids"] for example in batch])
        batch_attention_mask = torch.stack([example["attention_mask"] for example in batch])

        # Collect other fields as lists
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

    test_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        inputs = {
            key: val
            for key, val in batch.items()
            if key in ["input_ids", "attention_mask"]
        }
        # Move inputs to the correct device
        inputs["input_ids"] = inputs["input_ids"]  # .to(device)
        inputs["attention_mask"] = inputs["attention_mask"]  # .to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_length:]
        predicted_sqls = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        batch_prediction_scores = []
        batch_gt_scores = []
        for i in range(len(predicted_sqls)):
            predicted_sql = predicted_sqls[i].strip()
            # TODO: Make this more efficient
            item_batch = {
                key: value[i] if isinstance(value, (list, torch.Tensor)) else value
                for key, value in batch.items()
                if key not in ['input_ids', 'attention_mask']
            }
            #import ipdb; ipdb.set_trace()
            item = dict_to_batchitem(item_batch)
            prediction_score = evaluate_cs_function(
                item, predicted_sql
            )
            gt_score = evaluate_cs_function(
                item, item.sql_statement
            )
            batch_prediction_scores.append(prediction_score)
            batch_gt_scores.append(gt_score)

            # if args.debug and (prediction_score < 1 or prediction_score < gt_score):
            #     print("Table:", table_names[i])
            #     print("Selected fields:", selected_fields_list[i])
            #     print("English prompt:", english_prompts[i])
            #     print("Create statement:", create_statements[i])
            #     print("SQL (Ground Truth):", sql_statements[i])
            #     print("SQL (Prediction):", predicted_sql)
            #     print("Score (Prediction):", prediction_score)
            #     print("Score (Ground Truth):", gt_score)

        batch_prediction_scores_tensor = torch.tensor(batch_prediction_scores)
        batch_gt_scores_tensor = torch.tensor(batch_gt_scores)
        total_prediction_score += batch_prediction_scores_tensor.sum().item()
        total_gt_score += batch_gt_scores_tensor.sum().item()
        num_samples += len(batch_prediction_scores)

    gt_accuracy = (total_gt_score / num_samples) * 100 if num_samples > 0 else 0
    prediction_accuracy = (
        (total_prediction_score / num_samples) * 100 if num_samples > 0 else 0
    )
    # Log the scores to W&B with the dataset type
    wandb.log(
        {
            f"{dataset_type.capitalize()} Ground Truth Accuracy": gt_accuracy,
            f"{dataset_type.capitalize()} Prediction Accuracy": prediction_accuracy,
        }
    )

    print(f"{dataset_type.capitalize()} Ground Truth Accuracy: {gt_accuracy}")
    print(f"{dataset_type.capitalize()} Prediction Accuracy: {prediction_accuracy}")
    return gt_accuracy, prediction_accuracy


def push_model_to_hf(
    output_dir,
    repo_name,
    organization,
    commit_message="Add final model and tokenizer files",
    private=True,
):
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.push_to_hub(
        repo_id=f"{organization}/{repo_name}",
        commit_message=commit_message,
        private=private,
    )
    tokenizer.push_to_hub(
        repo_id=f"{organization}/{repo_name}", commit_message=commit_message
    )
    print(
        f"Model and tokenizer pushed to Hugging Face Hub at {organization}/{repo_name}"
    )


def sft(args):
    wandb.init(
        project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity
    )
    try:
        # Decide evaluation function based on the dataset
        # withmartian/cs1_dataset
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
        test_dataset = load_dataset(args.dataset_name, split="test")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, use_auth_token=args.hf_token
        )
        tokenizer.model_max_length = args.max_seq_length
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            use_auth_token=args.hf_token,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.config.attn_implementation = "flash_attention_2"

        alpaca_prompt = """### Instruction:\n{}\n### Context:\n{}\n### Response:\n"""

        def preprocess_function(examples):
            prompts = [
                alpaca_prompt.format(instruction, context)
                for instruction, context in zip(
                    examples["english_prompt"], examples["create_statement"]
                )
            ]
            responses = [sql + tokenizer.eos_token for sql in examples["sql_statement"]]
            prompt_encodings = tokenizer(
                prompts,
                truncation=True,
                max_length=args.max_seq_length,
                add_special_tokens=True,
            )
            response_encodings = tokenizer(
                responses,
                truncation=True,
                max_length=args.max_seq_length,
                add_special_tokens=False,
            )

            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for prompt_ids, response_ids in zip(
                prompt_encodings["input_ids"], response_encodings["input_ids"]
            ):
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

        # if args.debug:
        #     train_dataset = train_dataset.select(range(12))
        #     val_dataset = val_dataset.select(range(12))

        train_dataset_processed = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        val_dataset_processed = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

        num_training_examples = len(train_dataset_processed)
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        effective_batch_size = (
            args.batch_size * args.gradient_accumulation_steps
        )  # * num_devices
        steps_per_epoch = max(1, num_training_examples // effective_batch_size)
        eval_steps = steps_per_epoch // 16
        save_steps = steps_per_epoch // 2
        print(f"Steps per epoch: {steps_per_epoch}, Eval steps: {eval_steps}")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

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
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset_processed,
            eval_dataset=val_dataset_processed,
            data_collator=data_collator,
            max_seq_length=args.max_seq_length,
            args=training_args,
        )

        # Create the evaluation callback
        evaluation_callback = EvaluationCallback(
            evaluate_fn=evaluate,
            trainer=trainer,
            dataset=val_dataset,
            model=model,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            evaluate_cs_function=evaluate_cs_function,
            dataset_type="validation",
        )
        trainer.add_callback(evaluation_callback)

        # Evaluate the model before fine-tuning
        # evaluate(
        #     args,
        #     val_dataset,
        #     model,
        #     tokenizer,
        #     args.max_seq_length,
        #     evaluate_cs_function,
        #     dataset_type="validation",
        # )
        # evaluate(
        #     args,
        #     test_dataset,
        #     model,
        #     tokenizer,
        #     args.max_seq_length,
        #     evaluate_cs_function,
        #     dataset_type="test",
        # )

        trainer.train()

        # Evaluate the model after fine-tuning
        evaluate(
            args,
            val_dataset,
            model,
            tokenizer,
            args.max_seq_length,
            evaluate_cs_function,
            dataset_type="validation",
        )
        evaluate(
            args,
            test_dataset,
            model,
            tokenizer,
            args.max_seq_length,
            evaluate_cs_function,
            dataset_type="test",
        )

        unwrapped_model = model
        unwrapped_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model and tokenizer saved in {args.output_dir}")
        push_model_to_hf(args.output_dir, args.wandb_run_name, "withmartian")

    finally:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    if args.sft:
        sft(args)
    if args.evaluate:
        evaluate(args)

    if not args.sft and not args.evaluate:
        print("Please specify either --sft, --evaluate, or both")
        sys.exit(1)
