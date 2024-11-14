import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from QuantaTextToSql.training_data import generate_cs1
from QuantaTextToSql.training_data.generate_utils import generate_inputs_from_prompt, generate_inputs_from_BatchItems, output_inference_text_with_hook


# Load the tokenizer and base model from Hugging Face
def load_tm1():
    auth_token = os.getenv("HF_AUTH_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        "withmartian/sft_sql_interp_TinyStories-33M_cs1_experiment_7.3",
        token=auth_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        "withmartian/sft_sql_interp_TinyStories-33M_cs1_experiment_7.3",
        token=auth_token
    )
    
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
        
    return tokenizer, model


# Dictionary to store average activations for all layers, heads, and MLPs
average_tm1_activations = {
    "head": {},  # Dict of layer -> list of average activations per head
    "mlp": [],   # List of MLP average activations
    "layer": []  # List of layer average activations
}


# Function to collect average activations for all heads, MLPs, and layers
def collect_tm1_activations(model, tokenizer):
    #inputs = generate_inputs_from_prompt(tokenizer)
    batch_items = generate_cs1(317)
    inputs = generate_inputs_from_BatchItems(tokenizer, batch_items)

    # Hook to collect average activations
    def collect_activations_hook(module, input, output, layer_index=None, head_index=None):
        if isinstance(output, tuple):
            output = output[0]
        
        if layer_index is not None:
            if head_index is not None:
                # Store average activation for a specific attention head
                if layer_index not in average_tm1_activations["head"]:
                    average_tm1_activations["head"][layer_index] = []
                while len(average_tm1_activations["head"][layer_index]) <= head_index:
                    average_tm1_activations["head"][layer_index].append(None)
                head_activation = output[:, :, head_index].mean(dim=1).detach().clone()
                average_tm1_activations["head"][layer_index][head_index] = head_activation
            else:
                # Store average activation for the entire layer
                average_tm1_activations["layer"].append(output.mean(dim=1).detach().clone())

    num_heads = model.config.num_attention_heads
    
    # Register hooks for all layers
    for layer_index, layer in enumerate(model.transformer.h):
        # Collect layer-wide activations
        layer.register_forward_hook(lambda m, i, o, li=layer_index: collect_activations_hook(m, i, o, layer_index=li))
        
        # Collect attention head activations
        attention_module = model.transformer.h[layer_index].attn        
        for head_index in range(num_heads):
            attention_module.register_forward_hook(lambda m, i, o, li=layer_index, hi=head_index:
                                                       collect_activations_hook(m, i, o, layer_index=li, head_index=hi))
        
        # Collect MLP activations by averaging the MLP output directly
        def collect_mlp_output(module, input, output, li=layer_index):
            mlp_activation = output.mean(dim=0).detach().clone()
            average_tm1_activations["mlp"].append(mlp_activation)

        
        layer.mlp.register_forward_hook(collect_mlp_output)
    
    # Run forward pass to collect all average activations
    model(**inputs)


# Function to ablate using pre-collected average activations
def ablate_tm1(tokenizer, model, node_type="layer", layer_index=0, head_index=None):
    inputs = generate_inputs_from_prompt(tokenizer)
    
    # Ablation hook that uses pre-stored average values
    def ablation_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        
        # Replace output with average activation depending on node type
        if node_type == "head" and head_index is not None:
            output[:, :, head_index] = average_tm1_activations["head"][layer_index][head_index]
        elif node_type == "mlp":
            # Adjust stored MLP activation to match `output`'s sequence length
            the_activation = average_tm1_activations["mlp"][layer_index].clone()
            the_activation = the_activation.unsqueeze(0) 
 
            # Not sure why we need to do this to reduce dim1 from 191 to 13
            if the_activation.size(1) != output.size(1):
               the_activation = the_activation[:, :output.size(1), :]
            
            output[:] = the_activation
        elif node_type == "layer":
            the_activation = average_tm1_activations["layer"][layer_index].clone()
            the_activation = the_activation.unsqueeze(0) 

            # Not sure why we need to do this to reduce dim1 from 317 to 13
            if the_activation.size(1) != output.size(1):
               the_activation = the_activation[:, :output.size(1), :]

            output[:] = the_activation
    
    # Register ablation hook
    layer = model.transformer.h[layer_index]
    ablation_hook = layer.register_forward_hook(ablation_hook)
    
    # Generate text with ablated nodes
    output_inference_text_with_hook(tokenizer, model, inputs)
    
    # Remove the ablation hook after use
    ablation_hook.remove()
