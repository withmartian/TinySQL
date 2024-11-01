from dataclasses import dataclass
from typing import List
import pandas as pd
from datasets import Dataset, DatasetDict

import sys
sys.path.append('/mnt/foundation-shared/dhruv_gretel_ai/research/sql/quanta_text_to_sql')
from training_data import get_sql_table_names, get_sql_field_names, get_sql_create_table, get_sql_select_from, get_english_select_from_phrase


@dataclass
class BatchItem:
    table_name: str
    table_fields: List[str]
    create_statement: str
    selected_fields: List[str]
    english_prompt: str
    sql_statement: str


# Generate a batch of "command set 1" prompts and answers. These are SQL SELECT statements with a single table and a few fields.
def generate_cs1(batch_size):
    table_names = get_sql_table_names()
    field_names_and_types = get_sql_field_names()       

    batch = []
    for i in range(batch_size):
        (table_name, table_fields, create_table_statement) = get_sql_create_table(table_names, field_names_and_types, 2, 12)

        (selected_fields, sql_select_statement) = get_sql_select_from(table_name, table_fields)

        english_select_from_prompt = get_english_select_from_phrase(table_name, selected_fields)

        batch_item = BatchItem(
            table_name=table_name,
            table_fields=table_fields,
            create_statement=create_table_statement,
            selected_fields=selected_fields,
            english_prompt=english_select_from_prompt,
            sql_statement=sql_select_statement,
        )

        batch.append(batch_item)

    return batch


# Returns accuracy of a "command set 1" predicted answer compared to the ground truth. Accuracy is in range 0 to 1. 
def evaluate_cs1_prediction(table_name, selected_fields, predicted_sql_statement):

    # We want to reward components of the answer that are correct.
    # - Contains SELECT                     1 point
    # - Contains table_name                 1 point
    # - Contains FROM                       1 point
    # - Contains selected field names       N points
    # - First word is SELECT                1 point
    # - All field names are after SELECT    N points
    # - Word FROM is after all field names  1 point
    # - Word table_name is after FROM       1 point   
    # - There are no unrecognished words    1 point


    # Calculate the total number of points on offer
    N = len(selected_fields)
    total_points = 7 + 2 * N  # 7 fixed points + 2 points per field


    points_earned = 0

    # Tokenize the predicted SQL statement
    tokens = predicted_sql_statement.strip().split()
    tokens_upper = [token.upper().strip(',') for token in tokens]

    # Criterion 1: Contains SELECT (1 point)
    if 'SELECT' in tokens_upper:
        points_earned += 1

    # Criterion 2: Contains table_name (1 point)
    if table_name.upper() in tokens_upper:
        points_earned += 1

    # Criterion 3: Contains FROM (1 point)
    if 'FROM' in tokens_upper:
        points_earned += 1

    # Criterion 4: Contains field names (N points)
    field_points = 0
    for field in selected_fields:
        if field.upper() in tokens_upper:
            field_points += 1
    points_earned += field_points

    # Criterion 5: First word is SELECT (1 point)
    if len(tokens_upper) > 0 and tokens_upper[0] == 'SELECT':
        points_earned += 1

    # Criterion 6: All field names are after SELECT (N points)
    field_after_select_points = 0
    if 'SELECT' in tokens_upper:
        select_index = tokens_upper.index('SELECT')
        for field in selected_fields:
            field_indices = [i for i, token in enumerate(tokens_upper) if token == field.upper()]
            if all(i > select_index for i in field_indices):
                field_after_select_points += 1
    points_earned += field_after_select_points

    # Criterion 7: Word FROM is after all field names (1 point)
    if 'FROM' in tokens_upper:
        from_index = tokens_upper.index('FROM')
        last_field_index = max(
            [i for i, token in enumerate(tokens_upper) if token in [f.upper() for f in selected_fields]],
            default=-1
        )
        if from_index > last_field_index:
            points_earned += 1

    # Criterion 8: table_name is after FROM (1 point)
    if 'FROM' in tokens_upper and table_name.upper() in tokens_upper:
        from_index = tokens_upper.index('FROM')
        table_name_index = tokens_upper.index(table_name.upper())
        if table_name_index > from_index:
            points_earned += 1

    # Criterion 9: There are no unrecognized words (1 point)
    recognized_words = ['SELECT', 'FROM'] + [table_name.upper()] + [field.upper() for field in selected_fields]
    unrecognized_words = [token for token in tokens_upper if token not in recognized_words]
    if len(unrecognized_words) == 0:
        points_earned += 1

    accuracy = 1.0 * points_earned / total_points
    return accuracy

# main function
if __name__ == '__main__':
    # 20s to generate 100k samples
    batch_size = 100000
    debug = False
    dataset = generate_cs1(batch_size)

    # Create a dataframe
    df = pd.DataFrame(dataset, columns=['english_prompt', 'create_statement', 'sql_statement', 'selected_fields', 'table_fields', 'table_name'])

    # Convert to HF dataset
    hf_dataset = Dataset.from_pandas(df)

    # Split dataset into train + val (90%) and test (10%)
    train_val_split = hf_dataset.train_test_split(test_size=0.1, seed=420)
    test_dataset = train_val_split['test']
    
    # Further split train_val into train (85%) and val (15% of the remaining data)
    train_val_split = train_val_split['train'].train_test_split(test_size=0.15, seed=420)  # 0.25 * 0.8 = 0.2

    # Organize splits
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']

    # Combine splits into a single DatasetDict
    hf_dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # Push to HF
    hf_dataset.push_to_hub("dhruvnathawani/cs1_dataset")

    # Unit tests for the dataset (on the train set as an example)
    accuracy = 0
    for i in range(len(train_dataset)):
        score = evaluate_cs1_prediction(dataset[i].table_name, dataset[i].selected_fields, dataset[i].sql_statement)
        if debug or score < 1:
            print("Table:", dataset[i].table_name)
            print("Table fields:", dataset[i].table_fields)
            print("Create:", dataset[i].create_statement)
            print("Selected fields:", dataset[i].selected_fields)
            print("English:", dataset[i].english_prompt)
            print("SQL:", dataset[i].sql_statement)
        #assert(score == 1)
        accuracy += score
    print("Dataset Accuracy:", (accuracy / len(train_dataset)) * 100)
