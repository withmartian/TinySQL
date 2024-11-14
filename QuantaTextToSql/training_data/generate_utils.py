

def generate_inputs_from_prompt(tokenizer, prompt_text="Once upon a time, in a small village, there was a"):
    inputs = tokenizer(prompt_text, padding=True, return_tensors="pt")  # Pad to ensure matching dimensions
    return inputs

def generate_inputs_from_BatchItems(tokenizer, batch_items):

    prompt_texts = [batch_item.get_alpaca_prompt() for batch_item in batch_items]

    # Tokenize with padding to the longest sequence
    batched_inputs = tokenizer(prompt_texts, padding=True, return_tensors="pt")

    return (prompt_texts, batched_inputs)

def output_inference_text(tokenizer, outputs):
    next_token_logits = outputs.logits[:, -1, :]
    predicted_token_id = next_token_logits.argmax(-1)
    inference_text = tokenizer.decode(predicted_token_id, skip_special_tokens=True)
    return inference_text
