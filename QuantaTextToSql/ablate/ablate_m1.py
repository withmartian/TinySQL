import torch
from QuantaTextToSql.training_data.generate_utils import output_inference_text


# Function to collect average activations for all heads, MLPs, and layers
def collect_m1_activations(model, batched_inputs):

    # Dictionary to store average activations for all layers, heads, and MLPs
    cached_avg_activations = {
        "head": {},  # Dict of layer -> list of average activations per head
        "mlp": [],   # List of MLP average activations
        "layer": []  # List of layer average activations
    }

    # Hook to collect average activations
    def collect_activations_hook(module, input, output, layer_index=None, head_index=None):
        if isinstance(output, tuple):
            output = output[0]
        
        if layer_index is not None:
            if head_index is not None:
                # Store average activation for a specific attention head from output[BatchSize=100, SeqLen=183, HiddenDimension=1024]
                if layer_index not in cached_avg_activations["head"]:
                    cached_avg_activations["head"][layer_index] = []
                while len(cached_avg_activations["head"][layer_index]) <= head_index:
                    cached_avg_activations["head"][layer_index].append(None)
                head_activation = output[:, :, head_index].mean(dim=0).detach().clone() # Gives head_activation[SeqLen=183]
                cached_avg_activations["head"][layer_index][head_index] = head_activation
            else:
                # Store average activation for the entire layer
                cached_avg_activations["layer"].append(output.mean(dim=0).detach().clone())

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
            mlp_activation = output.mean(dim=0).detach().clone() # gives size [SeqLen=183, HiddenDimension=1024]
            #print(f"collect_bm1: output.shape: {output.shape}, mlp_activation.shape: {mlp_activation.shape}") 
            cached_avg_activations["mlp"].append(mlp_activation)

        
        layer.mlp.register_forward_hook(collect_mlp_output)
    
    # Run inference (forward pass) without ablation to collect all average activations
    model(**batched_inputs)

    return cached_avg_activations


# Function to ablate using pre-collected average activations
def ablated_m1_inference(tokenizer, model, cached_avg_activations, batched_inputs, node_type="layer", layer_index=0, head_index=None):
    
    # Ablation hook that uses pre-stored average values
    def ablation_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        
        # Replace output with average activation depending on node type
        if node_type == "head" and head_index is not None:
            the_activation = cached_avg_activations["head"][layer_index][head_index].clone()
            the_activation = the_activation.repeat(output.size(0), 1)
            
            #print(f"ablate_head: output.shape: {output.shape}, the_activation.shape: {the_activation.shape}")
            output[:, :, head_index] = the_activation

        elif node_type == "mlp":
            the_activation = cached_avg_activations["mlp"][layer_index].clone()
            the_activation = the_activation.unsqueeze(0)

            # Adjust both sequence length (dim 1) and batch size (dim 0)
            if the_activation.size(1) != output.size(1):
                # Truncate or pad to match output sequence length
                if the_activation.size(1) > output.size(1):
                    the_activation = the_activation[:, :output.size(1), :]
                else:
                    # If activation is shorter, pad with zeros
                    pad_size = output.size(1) - the_activation.size(1)
                    the_activation = torch.nn.functional.pad(the_activation, (0, 0, 0, pad_size, 0, 0))
            the_activation = the_activation.expand(output.size(0), -1, -1)

            assert the_activation.shape == output.shape, f"Shape mismatch: activation {the_activation.shape} vs output {output.shape}"
            output[:] = the_activation

        elif node_type == "layer":
            the_activation = cached_avg_activations["layer"][layer_index].clone()
            the_activation = the_activation.unsqueeze(0)

            # Adjust both sequence length (dim 1) and batch size (dim 0)
            if the_activation.size(1) != output.size(1):
                # Truncate or pad to match output sequence length
                if the_activation.size(1) > output.size(1):
                    the_activation = the_activation[:, :output.size(1), :]
                else:
                    # If activation is shorter, pad with zeros
                    pad_size = output.size(1) - the_activation.size(1)
                    the_activation = torch.nn.functional.pad(the_activation, (0, 0, 0, pad_size, 0, 0))
            the_activation = the_activation.expand(output.size(0), -1, -1)

            assert the_activation.shape == output.shape, f"Shape mismatch: activation {the_activation.shape} vs output {output.shape}"
            output[:] = the_activation
    
    # Register ablation hook
    layer = model.transformer.h[layer_index]
    ablation_hook = layer.register_forward_hook(ablation_hook)

    # Run inference (forward pass) with ablation to understand effect
    outputs = model(**batched_inputs)  

    # Generate text with ablated nodes
    generated_text = output_inference_text(tokenizer, outputs)
    
    ablation_hook.remove()

    return (outputs, generated_text)
