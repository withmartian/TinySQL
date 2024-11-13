from transformers import AutoTokenizer, AutoModelForCausalLM
from QuantaTextToSql.training_data import generate_cs1


def generate_inputs_from_prompt(tokenizer, prompt_text="Once upon a time, in a small village, there was a"):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    return inputs

def generate_inputs_from_BatchItem(tokenizer, batch_item):
    prompt_text = batch_item.get_alpaca_prompt()
    inputs = tokenizer(prompt_text, return_tensors="pt")
    return inputs

def generate_inputs_from_BatchItems(tokenizer, batch_items):
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # or you can add a custom pad token

    # Collect all prompt texts
    prompt_texts = [batch_item.get_alpaca_prompt() for batch_item in batch_items]
    
    # Tokenize with padding to the longest sequence
    batched_inputs = tokenizer(prompt_texts, padding=True, return_tensors="pt")

    return batched_inputs



def output_inference_text(tokenizer, title, outputs):
    next_token_logits = outputs.logits[:, -1, :]
    predicted_token_id = next_token_logits.argmax(-1)
    generated_text = tokenizer.decode(predicted_token_id, skip_special_tokens=True)
    print(f"{title}: {generated_text}")

def output_inference_text_no_hook(tokenizer, model, inputs):
    output_inference_text(tokenizer, "Generated Text without Hook", model(**inputs))

def output_inference_text_with_hook(tokenizer, model, inputs):
    output_inference_text(tokenizer, "Generated Text with Hook", model(**inputs))


# Load the tokenizer and base model from Hugging Face
def load_bm1():
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-Instruct-2Layers-33M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-Instruct-2Layers-33M")
    return tokenizer, model


# Dictionary to store average activations for all layers, heads, and MLPs
average_bm1_activations = {
    "head": {},  # Dict of layer -> list of average activations per head
    "mlp": [],   # List of MLP average activations
    "layer": []  # List of layer average activations
}


# Function to collect average activations for all heads, MLPs, and layers
def collect_bm1_activations(model, tokenizer):
    #inputs = generate_inputs_from_prompt(tokenizer)
    batch_items = generate_cs1(100)
    inputs = generate_inputs_from_BatchItems(tokenizer, batch_items)

    # Hook to collect average activations
    def collect_activations_hook(module, input, output, layer_index=None, head_index=None):
        if isinstance(output, tuple):
            output = output[0]
        
        if layer_index is not None:
            if head_index is not None:
                # Store average activation for a specific attention head
                if layer_index not in average_bm1_activations["head"]:
                    average_bm1_activations["head"][layer_index] = []
                while len(average_bm1_activations["head"][layer_index]) <= head_index:
                    average_bm1_activations["head"][layer_index].append(None)
                head_activation = output[:, :, head_index].mean(dim=1).detach().clone()
                average_bm1_activations["head"][layer_index][head_index] = head_activation
            else:
                # Store average activation for the entire layer
                average_bm1_activations["layer"].append(output.mean(dim=1).detach().clone())

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
            mlp_activation = output.mean(dim=1).detach().clone()
            average_bm1_activations["mlp"].append(mlp_activation)
        
        layer.mlp.register_forward_hook(collect_mlp_output)
    
    # Run forward pass to collect all average activations
    model(**inputs)


# Function to ablate using pre-collected average activations
def ablate_bm1(tokenizer, model, node_type="layer", layer_index=0, head_index=None):
    inputs = generate_inputs_from_prompt(tokenizer)
    
    # Ablation hook that uses pre-stored average values
    def ablation_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        
        # Replace output with average activation depending on node type
        if node_type == "head" and head_index is not None:
            output[:, :, head_index] = average_bm1_activations["head"][layer_index][head_index]
        elif node_type == "mlp":
            output[:] = average_bm1_activations["mlp"][layer_index]
        elif node_type == "layer":
            output[:] = average_bm1_activations["layer"][layer_index]
    
    # Register ablation hook
    layer = model.transformer.h[layer_index]
    ablation_hook = layer.register_forward_hook(ablation_hook)
    
    # Generate text with ablated nodes
    output_inference_text_with_hook(tokenizer, model, inputs)
    
    # Remove the ablation hook after use
    ablation_hook.remove()
