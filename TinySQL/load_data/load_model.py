import gc
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel


# Load the tokenizer and trained model for model 1, 2, or 3 and command set 0 (base model), 1, 2, or 3
# If you are changing the models, consider updating the HF collection withmartian/tinysql as well. 
def sql_interp_model_location( model_num : int, cs_num : int):

    if model_num == 0:
        # Used with nnsight tutorials only
        return "openai-community/gpt2" # Base model

    elif model_num == 1:
        if cs_num == 0:
            return "roneneldan/TinyStories-Instruct-2Layers-33M"

        elif cs_num == 1:
            return "withmartian/sql_interp_bm1_cs1_experiment_1.8"
 
        elif cs_num == 2:
            return "withmartian/sql_interp_bm1_cs2_experiment_2.8"

        elif cs_num == 3:
            return "withmartian/sql_interp_bm1_cs3_experiment_3.8"
        
    elif model_num == 2:
        if cs_num == 0:
            return "Qwen/Qwen2.5-0.5B-Instruct"

        elif cs_num == 1:
            return "withmartian/sql_interp_bm2_cs1_experiment_4.1"
 
        elif cs_num == 2:
            return "withmartian/sql_interp_bm2_cs2_experiment_5.1"

        elif cs_num == 3:
            return "withmartian/sql_interp_bm2_cs3_experiment_6.1"
        
    elif model_num == 3:
        if cs_num == 0:
            return "withmartian/Llama-3.2-1B-Instruct"

        elif cs_num == 1:
            return "withmartian/sql_interp_bm3_cs1_experiment_7.1"
 
        elif cs_num == 2:
            return "withmartian/sql_interp_bm3_cs2_experiment_8.1"

        elif cs_num == 3:
            return "withmartian/sql_interp_bm3_cs3_experiment_9.1"
        
    elif model_num == 4: # draft
        return "ibm-granite/granite-3.0-1b-a400m-instruct" # Base "mid-size" model
    
    elif model_num == 5: # draft
        return "HuggingFaceTB/SmolLM-360M-Instruct" # Base "mid-size" model    


# Load the tokenizer and model. Uses HF_TOKEN for private models 
def load_model(model_location, auth_token=None, use_flash_attention=True, device_map="auto"):
    if auth_token is None:
        auth_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_location, 
        token=auth_token)
   
    if use_flash_attention:
        # qwen model and llama model with flash attention
        # Prerequisite: pip install flash-attn==2.0.2
        # From https://github.com/Dao-AILab/flash-attention
        auto_model = AutoModelForCausalLM.from_pretrained(
            model_location,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )
    else:
        # model without flash attention
        auto_model = AutoModelForCausalLM.from_pretrained(
            model_location,
            torch_dtype=torch.float32,
            device_map=device_map,
        )

    return tokenizer, auto_model


def load_sql_interp_model( model_num : int, cs_num : int, auth_token=None, use_flash_attention=True, device_map="auto"):
    model_location = sql_interp_model_location(model_num, cs_num)

    tokenizer, auto_model = load_model(model_location, auth_token=auth_token, use_flash_attention=use_flash_attention, device_map=device_map)

    if model_num == 1:
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        auto_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        auto_model.config.pad_token_id = tokenizer.pad_token_id

        auto_model.resize_token_embeddings(len(tokenizer))

    return tokenizer, auto_model


def load_tinysql_model( model_num : int, cs_num : int, auth_token=None):

    if model_num == 1:
        the_tokenizer, auto_model = load_sql_interp_model(model_num, cs_num, auth_token=auth_token, use_flash_attention=False)
        language_model = LanguageModel(auto_model, the_tokenizer)
        language_model.tokenizer = the_tokenizer
    else:
        language_model = LanguageModel(sql_interp_model_location(model_num, cs_num), device_map="auto")

    return language_model


# Free up memory. Deletes objects that only have "weak references" to them.
def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # If you are NOT working with gradients, you can use the following code, before a "trace" call, to free up memory.
    # with torch.no_grad():


# A list may contain 'weak references' to objects that are garbage collected by free_memory.
# This function replaces weak with strong references. Call it before free_memory. 
def replace_weak_references(obj):
    if isinstance(obj, list):
        return [replace_weak_references(item) for item in obj]
    elif hasattr(obj, 'value'):  # For objects with a 'value' attribute
        return obj.value
    elif hasattr(obj, 'get_value'):  # For objects with a 'get_value()' method
        return obj.get_value()
    elif isinstance(obj, torch.Tensor):
        return obj.clone().detach()
    elif hasattr(obj, 'item'):  # For objects with a 'item()' method
        return obj.item()
    else:
        return obj
    

def get_model_sizes( model_num, model, show = True ):
    # Old code
    # n_heads = 16 if model_num == 1 else 7 if model_num == 2 else 16
    # n_dim = 64 if model_num == 1 else 128 if model_num == 2 else 128  
        
    if model_num == 1:
        N_LAYERS = len(model.transformer.h)
    else:
        N_LAYERS = len(model.model.layers)
    N_HEADS = 16 if model_num == 1 else 7 if model_num == 2 else 16

    D_MODEL = model.transformer.wte.embedding_dim if model_num == 1 else model.config.hidden_size
    D_HEAD = D_MODEL // N_HEADS  

    if show:
        print("N_LAYERS="+str(N_LAYERS), "N_HEADS="+str(N_HEADS), "D_MODEL="+str(D_MODEL), "D_HEAD="+str(D_HEAD))

    return N_LAYERS, N_HEADS, D_MODEL, D_HEAD