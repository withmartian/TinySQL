def evaluate(args, dataset, model, tokenizer, max_seq_length, accelerator, data_collator, batch_size=1024):
    model.eval()
    total_prediction_score = 0
    total_gt_score = 0
    num_samples = 0

    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Set tokenizer padding side and special tokens
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token if pad token is undefined

    # Verify and set special token IDs
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|pad|>')

    # Tokenize the dataset outside the loop
    def tokenize_fn(example):
        prompt_text = f"### Instruction:\n{example['english_prompt']}\n### Context:\n{example['create_statement']}\n### Response:\n"
        return tokenizer(prompt_text, padding="max_length", truncation=True, max_length=max_seq_length)
    
    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    if args.debug:
        decoded_string = tokenizer.decode(
            [id for id, mask in zip(tokenized_dataset['input_ids'][0], tokenized_dataset['attention_mask'][0]) if mask == 1],
            skip_special_tokens=True
        )
        print("Decoded string:", decoded_string)

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Create a DataLoader for raw text input prompts for the evaluation dataset
    test_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
    #test_dataloader = accelerator.prepare(test_dataloader)

    # Move the model to the device before the loop
    #model = model.to(accelerator.device)

    # Loop over the DataLoader batches
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        # Move inputs to the appropriate device using accelerator
        inputs = {key: val.to(accelerator.device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}

        # Generate model output with no_grad for evaluation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        # Collect predicted SQL statements
        predicted_sqls = []
        for i in range(len(outputs)):
            input_ids = inputs['input_ids'][i]
            output_ids = outputs[i]

            # The total length of the input_ids
            input_length = input_ids.size(0)

            # Extract generated tokens by skipping the entire input_ids
            generated_ids = output_ids[input_length:]

            # Decode the generated tokens
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_sqls.append(generated_text.strip())

            # For debugging
            if args.debug:
                input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                print(f"\nInput Text {i}:\n{input_text}")
                print(f"Generated Output {i}:\n{generated_text}")

        # Evaluate each prediction and accumulate the score
        batch_prediction_scores = []
        batch_gt_scores = []
        for i, predicted_sql in enumerate(predicted_sqls):
            # Access the necessary fields from the batch
            table_name = dataset[i]['table_name']
            selected_fields = dataset[i]['selected_fields']
            sql_statement = dataset[i]['sql_statement']
            english_prompt = dataset[i]['english_prompt']
            create_statement = dataset[i]['create_statement']

            # Evaluate the prediction
            prediction_score = evaluate_cs1_prediction(
                table_name,
                selected_fields,
                predicted_sql
            )
            gt_score = evaluate_cs1_prediction(
                table_name,
                selected_fields,
                sql_statement
            )
            batch_prediction_scores.append(prediction_score)
            batch_gt_scores.append(gt_score)

            # Debug information for incorrect predictions
            if args.debug and (prediction_score < 1 or prediction_score < gt_score):
                print("Table:", table_name)
                print("Selected fields:", selected_fields)
                print("English prompt:", english_prompt)
                print("Create statement:", create_statement)
                print("SQL (Ground Truth):", sql_statement)
                print("SQL (Prediction):", predicted_sql)
                print("Score (Prediction):", prediction_score)
                print("Score (Ground Truth):", gt_score)

        # Use accelerator to gather scores across devices
        batch_prediction_scores_tensor = torch.tensor(batch_prediction_scores, device=accelerator.device)
        batch_gt_scores_tensor = torch.tensor(batch_gt_scores, device=accelerator.device)
        gathered_prediction_scores = accelerator.gather(batch_prediction_scores_tensor)
        gathered_gt_scores = accelerator.gather(batch_gt_scores_tensor)
        total_prediction_score += gathered_prediction_scores.sum().item()
        total_gt_score += gathered_gt_scores.sum().item()
        num_samples += len(gathered_prediction_scores)

    # Calculate the overall accuracy
    assert num_samples == len(dataset), "Number of samples processed should match the dataset size"
    gt_accuracy = (total_gt_score / num_samples) * 100 if num_samples > 0 else 0
    prediction_accuracy = (total_prediction_score / num_samples) * 100 if num_samples > 0 else 0
    print("Dataset Ground Truth Accuracy:", gt_accuracy)
    print("Model Prediction Accuracy:", prediction_accuracy)
    return gt_accuracy, prediction_accuracy