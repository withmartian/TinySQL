import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def refine_tokenizer(tokenizer):
    # Set the padding side
    tokenizer.padding_side = "left"

    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    return tokenizer



# Load the tokenizer and base model from Hugging Face
def load_bm1():
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-Instruct-2Layers-33M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-Instruct-2Layers-33M")
    
    tokenizer = refine_tokenizer(tokenizer)
        
    return tokenizer, model


# Load the tokenizer and trained model from Hugging Face
def load_bm1_from_hf(hf_location):
    auth_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(hf_location, token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(hf_location, token=auth_token)
    
    tokenizer = refine_tokenizer(tokenizer)
        
    return tokenizer, model


# Load the tokenizer and trained model from Hugging Face
def load_bm1_cs1():
    return load_bm1_from_hf("withmartian/sql_interp_bm1_cs1_experiment_1.1")

def load_bm1_cs2():
    return load_bm1_from_hf("withmartian/sql_interp_bm1_cs2_experiment_2.3")

def load_bm1_cs3():
    return load_bm1_from_hf("withmartian/sql_interp_bm1_cs3_experiment_3.3")
