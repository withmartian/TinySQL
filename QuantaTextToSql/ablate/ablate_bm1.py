from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate_inputs_from_prompt(tokenizer):
    prompt_text = "Once upon a time, in a small village, there was a"
    inputs = tokenizer(prompt_text, return_tensors="pt")
    return inputs


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


def ablate_bm1_layer(tokenizer, model, layer_index): 

    inputs = generate_inputs_from_prompt(tokenizer)

    output_inference_text_no_hook(tokenizer, model, inputs)


    def hook_fn(module, input, output):
        # Ensure the output is a tuple
        if isinstance(output, tuple):
            # Zero out the attention output (first element of the tuple)
            modified_attn_output = output[0] * 0
            # Reconstruct the output tuple with modified attention output
            modified_output = (modified_attn_output,) + output[1:]
        else:
            # If output is not a tuple, apply modification directly
            modified_output = output * 0
        return modified_output


    target_module = model.transformer.h[layer_index].attn
    hook_handle = target_module.register_forward_hook(hook_fn)

    output_inference_text_with_hook(tokenizer, model, inputs)

    hook_handle.remove()


def ablate_bm1_mlp(tokenizer, model, layer_index): 

    inputs = generate_inputs_from_prompt(tokenizer)

    output_inference_text_no_hook(tokenizer, model, inputs)


    # Define the hook function for the MLP layer
    def hook_fn(module, input, output):
        # Modify the output tensor
        modified_output = output * 0  # Example: Zeroing out the MLP output
        return modified_output


    # Choose the target module (MLP layer of the first transformer block)
    target_module = model.transformer.h[layer_index].mlp
    hook_handle = target_module.register_forward_hook(hook_fn)

    output_inference_text_with_hook(tokenizer, model, inputs)

    hook_handle.remove()  


def ablate_bm1_head(cfg, tokenizer, model, layer_index, head_index): 

    inputs = generate_inputs_from_prompt(tokenizer)

    output_inference_text_no_hook(tokenizer, model, inputs)


    # Define the hook function
    def hook_fn(module, input, output):
        # output[0] is the attention output
        # output[0] shape: (batch_size, seq_length, hidden_size)
        # We need to manipulate the attention scores before they are combined

        # Access the attention scores before combining heads
        # In GPT-Neo, this is a bit involved because the attention heads are merged
        # We'll need to modify the internals of the attention module

        # However, since the output is after combining heads, we can attempt to modify the output
        # corresponding to the specific head

        # Unfortunately, PyTorch hooks do not provide direct access to internal variables like attention weights
        # So we need to modify the attention module to expose these variables, or use a custom forward method

        # Modify the query, key, and value weights of the specific head
        with torch.no_grad():
            head_size = len(output[0]) // cfg.n_heads
            start = head_index * head_size
            end = start + head_size

            # Zero out the output for the specific head
            output[0][:, :, start:end] = 0

        return output


    attention_module = model.transformer.h[layer_index].attn
    hook_handle = attention_module.register_forward_hook(hook_fn)

    output_inference_text_with_hook(tokenizer, model, inputs)

    hook_handle.remove()