

def generate_inputs_from_prompt(tokenizer, prompt_text="Once upon a time, in a small village, there was a"):
    inputs = tokenizer(prompt_text, padding=True, return_tensors="pt")  # Pad to ensure matching dimensions
    return inputs

def generate_inputs_from_BatchItems(tokenizer, batch_items):

    prompt_texts = [batch_item.get_alpaca_prompt() for batch_item in batch_items]

    # Tokenize with padding to the longest sequence
    batched_inputs = tokenizer(prompt_texts, padding=True, return_tensors="pt")

    return (prompt_texts, batched_inputs)

def output_inference_text(tokenizer, outputs):
    # Process logits for each item in the batch
    logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
    batch_size = logits.size(0)
    inference_texts = []

    for i in range(batch_size):
        next_token_logits = logits[i, -1, :]  # Shape: [vocab_size] for the last token of the i-th batch
        predicted_token_id = next_token_logits.argmax(-1)  # Get the predicted token ID
        inference_text = tokenizer.decode(predicted_token_id, skip_special_tokens=True)
        inference_texts.append(inference_text)  # Append the decoded string to the list

    return inference_texts  # Return a list of strings

