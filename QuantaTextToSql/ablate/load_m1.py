import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and trained tiny-stories model from Hugging Face
def load_bm1_from_hf(hf_location):
    auth_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(hf_location, token=auth_token)
   
    # model without flash attention
    model = AutoModelForCausalLM.from_pretrained(
            hf_location,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    model.config.pad_token_id = tokenizer.pad_token_id

    model.resize_token_embeddings(len(tokenizer))
        
    return tokenizer, model


def load_bm1():
    return load_bm1_from_hf("roneneldan/TinyStories-Instruct-2Layers-33M")

def load_bm1_cs1():
    return load_bm1_from_hf("withmartian/sql_interp_bm1_cs1_experiment_1.1")

def load_bm1_cs2():
    return load_bm1_from_hf("withmartian/sql_interp_bm1_cs2_experiment_2.3")

def load_bm1_cs3():
    return load_bm1_from_hf("withmartian/sql_interp_bm1_cs3_experiment_3.3")
