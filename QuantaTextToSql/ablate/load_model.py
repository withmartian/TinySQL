import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Load the tokenizer and trained model for model 1, 2, or 3 and command set 0 (base model), 1, 2, or 3
def sql_interp_model_location( model_num : int, cs_num : int):
    if model_num == 1:
        if cs_num == 0:
            return "roneneldan/TinyStories-Instruct-2Layers-33M"

        elif cs_num == 1:
            return "withmartian/sql_interp_bm1_cs1_experiment_1.1"
 
        elif cs_num == 2:
            return "withmartian/sql_interp_bm1_cs2_experiment_2.3"

        elif cs_num == 3:
            return "withmartian/sql_interp_bm1_cs3_experiment_3.3"
        
    if model_num == 2:
        if cs_num == 0:
            return "huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"

        elif cs_num == 1:
            return "withmartian/sql_interp_bm2_cs1_experiment_4.1"
 
        elif cs_num == 2:
            return "withmartian/sql_interp_bm2_cs2_experiment_5.1"

        elif cs_num == 3:
            return "withmartian/sql_interp_bm2_cs3_experiment_6.1"
        
    if model_num == 3:
        if cs_num == 0:
            return "huggingface.co/meta-llama/Llama-3.2-1B-Instruct"

        elif cs_num == 1:
            return "withmartian/sql_interp_bm3_cs1_experiment_7.1"
 
        elif cs_num == 2:
            return "withmartian/sql_interp_bm3_cs2_experiment_8.1"

        elif cs_num == 3:
            return "withmartian/sql_interp_bm3_cs3_experiment_9.1"

    return ""


# Load the tokenizer and model. Uses HF_TOKEN for private models 
def load_model(model_location):
    auth_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_location, token=auth_token)
   
    # model without flash attention
    model = AutoModelForCausalLM.from_pretrained(
            model_location,
            torch_dtype=torch.float32,
            device_map="auto",
        )
               
    return tokenizer, model


def load_sql_interp_model( model_num : int, cs_num : int):
    model_location = sql_interp_model_location(model_num, cs_num)

    tokenizer, model = load_model(model_location)

    if model_num == 1:
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        model.config.pad_token_id = tokenizer.pad_token_id

        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

