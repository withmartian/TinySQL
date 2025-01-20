import torch

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from TinySQL.training_data.generate_datasets import dict_to_batchitem
from TinySQL.load_data.load_model import sql_interp_model_location
from TinySQL.training_data.generate_cs1 import evaluate_cs1_prediction
from TinySQL.training_data.generate_cs2 import evaluate_cs2_prediction
from TinySQL.training_data.generate_cs3 import evaluate_cs3_prediction


def get_errors(max_seq_length=512, cs_num=3, model_num=1, syn=True, batch_size=32):

    model_name = sql_interp_model_location(model_num=model_num, cs_num=cs_num, synonym=syn)
    dataset_name = f"withmartian/cs{cs_num}_dataset_synonyms" if syn else f"withmartian/cs{cs_num}_dataset"

    alpaca_prompt = """### Instruction: {} ### Context: {} ### Response: """
    dataset_type = "validation"

    correct_predictions = []
    errors = []

    match str(cs_num):
        case "1":
            eval_function = evaluate_cs1_prediction
        case "2":
            eval_function = evaluate_cs2_prediction
        case "3":
            eval_function = evaluate_cs3_prediction


    print(f"Loading dataset {dataset_name}")

    dataset = load_dataset(dataset_name)[dataset_type].select(range(100))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    # Set the padding side
    tokenizer.padding_side = "left"

    errors = []
    correct_predictions = []

    # Loop over the dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {dataset_type} "):
        # Get the batch of samples as a dictionary of lists
        batch_samples_dict = dataset[i:i+batch_size]

        # Get the number of samples in the batch
        num_samples = len(next(iter(batch_samples_dict.values())))

        # Convert to a list of dictionaries
        batch_samples = [
            {key: batch_samples_dict[key][idx] for key in batch_samples_dict}
            for idx in range(num_samples)
        ]

        # Prepare prompts for each sample in the batch
        prompts = [
            alpaca_prompt.format(sample["english_prompt"], sample["create_statement"])
            for sample in batch_samples
        ]

        for sample in batch_samples:
            sample["full_output"] = alpaca_prompt.format(sample["english_prompt"], sample["create_statement"]) + " " + sample["sql_statement"]

        # Tokenize the prompts, use padding and truncation to handle variable-length inputs
        encoding = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )

        # Move input_ids and attention_mask to the model's device
        input_ids = encoding['input_ids'].to(model.device)
        attention_mask = encoding['attention_mask'].to(model.device)

        # Generate outputs using model.generate
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            temperature=0.5,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            early_stopping=True,
        )

        # Process each generated output in the batch
        for idx, sample in enumerate(batch_samples):
            # Get the generated tokens for this sample
            generated_id = generated_ids[idx]

            # The prompt length may vary due to padding, so we need to find where the prompt ends
            input_length = input_ids.shape[1]

            # Extract the generated tokens after the prompt
            generated_tokens = generated_id[input_length:]

            # Decode the generated tokens to get the predicted SQL
            generated_sql = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Prepare the item for evaluation
            item = dict_to_batchitem(sample)

            # Evaluate the predicted SQL
            prediction_score = eval_function(item, generated_sql)

            expected_sql = sample['sql_statement']
            english = sample['english_prompt']
            context = sample['create_statement']
            full_output = sample['full_output']

            local_dict = {
                'generated': generated_sql, 'expected': expected_sql,
                'context': context, 'english': english, 'full_output': full_output
            }

            # Update counters and sums
            if prediction_score == 1.00:
                correct_predictions.append(local_dict)
            else:
                errors.append(local_dict)

    data = {
        "errors": errors, "correct_predictions": correct_predictions
    }

    hf_data = DatasetDict({
        split_name: Dataset.from_list(split_data)
        for split_name, split_data in data.items()
    })

    return hf_data
