from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Load the dataset and tokenizer
dataset = load_dataset("dhruvnathawani/cs1_dataset")
#model_name = "meta-llama/Llama-3.2-3B-Instruct"
#TODO: Fix error with this model
#model_name = "Qwen/Qwen2-0.5B-Instruct-GGUF"
model_name = "roneneldan/TinyStories-Instruct-2Layers-33M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to compute token count per sample
def compute_token_counts(example):
    # Join all text fields to get the total token count per sample
    text = f"{example['english_prompt']} {example['create_statement']} {example['sql_statement']}"
    tokens = tokenizer(text, truncation=False)
    return {"token_count": len(tokens["input_ids"])}

# Apply the function to compute token counts
token_counts = dataset["train"].map(compute_token_counts)

# Calculate statistics
average_token_count = np.mean(token_counts["token_count"])
min_token_count = np.min(token_counts["token_count"])
max_token_count = np.max(token_counts["token_count"])

print(f"Average token count per sample: {average_token_count}")
print(f"Minimum token count per sample: {min_token_count}")
print(f"Maximum token count per sample: {max_token_count}")
