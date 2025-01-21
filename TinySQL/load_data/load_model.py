import gc
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from TinySQL.load_data.load_constants import BM_TUTORIAL, BM_TINY_STORIES, BM_QWEN, BM_LLAMA, BM_GRANITE, BM_SMOL, BM1_CS1_SEMANTIC, BM1_CS2_SEMANTIC, BM1_CS3_SEMANTIC, BM1_CS1_NONSEMANTIC, BM1_CS2_NONSEMANTIC, BM1_CS3_NONSEMANTIC, BM2_CS1_NONSEMANTIC, BM2_CS2_NONSEMANTIC, BM2_CS3_NONSEMANTIC, BM3_CS1_NONSEMANTIC, BM3_CS2_NONSEMANTIC, BM3_CS3_NONSEMANTIC    

# Load the tokenizer and trained model for model 1, 2, or 3 and command set 0 (base model), 1, 2, or 3
# If you are changing the models, consider updating the HF collection withmartian/tinysql as well. 
def sql_interp_model_location(model_num: int, cs_num: int, synonym: bool = True) -> str:
    """
    Get the model location based on model number, command set number, and synonym flag.
    
    Args:
        model_num: Model number (0-5)
        cs_num: Command set number (0-3)
        synonym: Whether to use semantic (True) or non-semantic (False) variants 
    
    Returns:
        str: Model location path
    
    Raises:
        ValueError: If invalid model_num or cs_num is provided
    """
    # Tutorial model
    if model_num == 0:
        return BM_TUTORIAL
        
    # Base models (cs_num == 0)
    if cs_num == 0:
        if model_num == 1:
            return BM_TINY_STORIES
        elif model_num == 2:
            return BM_QWEN
        elif model_num == 3:
            return BM_LLAMA
        elif model_num == 4:
            return BM_GRANITE
        elif model_num == 5:
            return BM_SMOL
            
    # Model 1 variants
    if model_num == 1:
        if synonym:  # Semantic variants
            if cs_num == 1:
                return BM1_CS1_SEMANTIC
            elif cs_num == 2:
                return BM1_CS2_SEMANTIC
            elif cs_num == 3:
                return BM1_CS3_SEMANTIC
        else:  # Non-semantic variants
            if cs_num == 1:
                return BM1_CS1_NONSEMANTIC
            elif cs_num == 2:
                return BM1_CS2_NONSEMANTIC
            elif cs_num == 3:
                return BM1_CS3_NONSEMANTIC
                
    # Model 2 variants (all non-semantic)
    elif model_num == 2:
        if cs_num == 1:
            return BM2_CS1_NONSEMANTIC
        elif cs_num == 2:
            return BM2_CS2_NONSEMANTIC
        elif cs_num == 3:
            return BM2_CS3_NONSEMANTIC
            
    # Model 3 variants (all non-semantic)
    elif model_num == 3:
        if cs_num == 1:
            return BM3_CS1_NONSEMANTIC
        elif cs_num == 2:
            return BM3_CS2_NONSEMANTIC
        elif cs_num == 3:
            return BM3_CS3_NONSEMANTIC
            
    raise ValueError(f"Invalid combination: model_num={model_num}, cs_num={cs_num}")
 


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


def load_sql_interp_model_location(model_num : int, model_location : str, auth_token=None, use_flash_attention=True, device_map="auto"):

    tokenizer, auto_model = load_model(model_location, auth_token=auth_token, use_flash_attention=use_flash_attention, device_map=device_map)

    if model_num == 1:
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        auto_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        auto_model.config.pad_token_id = tokenizer.pad_token_id

        auto_model.resize_token_embeddings(len(tokenizer))

    return tokenizer, auto_model


def load_sql_interp_model(model_num : int, cs_num : int, synonym : bool = True, auth_token=None, use_flash_attention=True, device_map="auto"):
    model_location = sql_interp_model_location(model_num, cs_num, synonym)

    return load_sql_interp_model_location(model_num, model_location, auth_token=auth_token, use_flash_attention=use_flash_attention, device_map=device_map)


def load_tinysql_model_location(model_num : int, model_location : str, auth_token=None):
    if model_num == 1:
        the_tokenizer, auto_model = load_sql_interp_model_location(model_num, model_location, auth_token=auth_token)
        language_model = LanguageModel(auto_model, the_tokenizer)
        language_model.tokenizer = the_tokenizer
    else:
        language_model = LanguageModel(model_location, device_map="auto")

    return language_model


def load_tinysql_model(model_num : int, cs_num : int, synonym : bool = True, auth_token=None):
    model_location = sql_interp_model_location(model_num, cs_num, synonym)

    return load_tinysql_model_location(model_num, model_location, auth_token=auth_token)


# Free up memory. 
# Deletes objects that only have 'weak references' to them. Refer comments on replace_weak_references below.
def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # If you are NOT working with gradients, you can use the following code, before a "trace" call, to free up memory.
    # with torch.no_grad():


# A list may contain 'weak references' to objects that are garbage collected by free_memory.
# This function replaces weak with strong references.
# If you get 'reference to deleted objects' when using nnsight, call this before calling free_memory. 
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
    

# Return key size information about the models. Only handles models 1 to 3 for now.
def get_model_sizes( model_num, model, show = True ):
        
    N_LAYERS = len(model.transformer.h) if model_num == 1 else model.model.layers
    N_HEADS = model.config.num_attention_heads

    D_MODEL = model.transformer.wte.embedding_dim if model_num == 1 else model.config.hidden_size
    D_HEAD = D_MODEL // N_HEADS  

    if show:
        print("N_LAYERS="+str(N_LAYERS), "N_HEADS="+str(N_HEADS), "D_MODEL="+str(D_MODEL), "D_HEAD="+str(D_HEAD))

    return N_LAYERS, N_HEADS, D_MODEL, D_HEAD