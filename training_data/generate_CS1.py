import random
from .fragments.english_aggregates import get_english_sum_phrases, get_english_avg_phrases, get_english_min_phrases, get_english_max_phrases, get_english_count_phrases
from .fragments.english_select_from import get_english_select_from_phrase
from .fragments.english_order_by import get_english_order_by_phrase
from .sql_create_table import get_sql_create_table
from .sql_select_from import get_sql_select_from
from .fragments.models import BatchItem, OrderField, SelectField

from dataclasses import dataclass
from typing import List
import pandas as pd
from datasets import Dataset, DatasetDict

import sys
sys.path.append('/mnt/foundation-shared/dhruv_gretel_ai/research/sql/quanta_text_to_sql')


def get_english_order_by(fields: list[OrderField]) -> str:
    answer = ""

    for i, field in enumerate(fields):
        english = get_english_order_by_phrase(field.asc)
        if i > 0:
            answer += ","
        answer += " " + english + " " + field.name
    
    return answer


def get_english_select_from(table_name: str, fields: list[SelectField]) -> str:
    template = get_english_select_from_phrase()    
    
    english_fields = ""
    for i, field in enumerate(fields):

        phrases = None
        if field.aggregate is None:
            # No aggregates  
            pass
        elif field.aggregate == "SUM":
            phrases = get_english_sum_phrases()
        elif field.aggregate == "AVG":
            phrases = get_english_avg_phrases()
        elif field.aggregate == "MIN":
            phrases = get_english_min_phrases()
        elif field.aggregate == "MAX":
            phrases = get_english_max_phrases()
        elif field.aggregate == "COUNT":
            phrases = get_english_count_phrases()

        if phrases is None:
            english_field = field.name
        else:
            english_field = f"{random.choice(phrases)} {field.name}"

        english_fields += english_field
        
        if i == len(fields) - 2:
            english_fields += " and "   
        elif i < len(fields) - 2:
            english_fields += ", "   
       
    # Create English phrase
    english = template.replace("[fields]", english_fields).replace("[table]", table_name)
    
    return english


# Generate a batch of "command set 1" prompts and answers. These are SQL SELECT statements with a single table and a few fields.
def generate_cs1(batch_size, min_cols=2, max_cols=12):

    batch = []
    for i in range(batch_size):
        (table_name, table_fields, create_table_statement) = get_sql_create_table(min_cols, max_cols)

        (selected_fields, sql_select_statement) = get_sql_select_from(table_name, table_fields, False)

        english_select_from_prompt = get_english_select_from(table_name, selected_fields)

        batch_item = BatchItem(
            command_set=1, 
            table_name=table_name,
            table_fields=table_fields,
            create_statement=create_table_statement,
            select=selected_fields,
            order_by=None,
            english_prompt=english_select_from_prompt,
            sql_statement=sql_select_statement, # ground truth
        )

        batch.append(batch_item)

    return batch


# Returns accuracy of a "command set 1" predicted answer compared to the ground truth
def evaluate_cs1_prediction_score_part1(item: BatchItem, predicted_sql_statement: str):

    # We want to reward components of the answer that are correct.
    # - Contains SELECT                     1 point
    # - Contains table_name                 1 point
    # - Contains FROM                       1 point
    # - First word is SELECT                1 point

    total_points = 0 
    points_earned = 0

    # Tokenize the predicted SQL statement
    tokens = predicted_sql_statement.strip().split()
    tokens_upper = [token.upper().strip(',') for token in tokens]

    # Criterion: Contains SELECT (1 point)
    total_points += 1
    if 'SELECT' in tokens_upper:
        points_earned += 1

    # Criterion: Contains table_name (1 point)
    total_points += 1
    if item.table_name.upper() in tokens_upper:
        points_earned += 1

    # Criterion: Contains FROM (1 point)
    total_points += 1
    if 'FROM' in tokens_upper:
        points_earned += 1

    # Criterion: First word is SELECT (1 point)
    total_points += 1
    if len(tokens_upper) > 0 and tokens_upper[0] == 'SELECT':
        points_earned += 1

    return (points_earned, total_points)


def evaluate_cs1_prediction_score_part2(item: BatchItem, predicted_sql_statement: str):

    # We want to reward components of the answer that are correct.
    # - Contains selected field names       N points    
    # - All field names are after SELECT    N points
    # - Word FROM is after all field names  1 point
    # - Word table_name is after FROM       1 point   
    # - There are no unrecognized words     1 point

    total_points = 0  
    points_earned = 0

    # Tokenize the predicted SQL statement
    tokens = predicted_sql_statement.strip().split()
    tokens_upper = [token.upper().strip(',') for token in tokens]

    # Criterion: Contains field names
    for field in item.select:
        total_points += 1
        if field.name.upper() in tokens_upper:
            points_earned += 1

    # Criterion: All field names are after SELECT 
    if 'SELECT' in tokens_upper:
        select_index = tokens_upper.index('SELECT')
        for field in item.select:
            field_indices = [i for i, token in enumerate(tokens_upper) if token == field.name.upper()]

            total_points += 1
            if all(i > select_index for i in field_indices):
                points_earned += 1

    # Criterion: Word FROM is after all field names 
    if 'FROM' in tokens_upper:
        from_index = tokens_upper.index('FROM')
        last_field_index = max(
            [i for i, token in enumerate(tokens_upper) if token in [f.name.upper() for f in item.select]],
            default=-1
        )
        total_points += 1
        if from_index > last_field_index:
            points_earned += 1

    # Criterion: table_name is after FROM 
    if 'FROM' in tokens_upper and item.table_name.upper() in tokens_upper:
        from_index = tokens_upper.index('FROM')
        table_name_index = tokens_upper.index(item.table_name.upper())

        total_points += 1
        if table_name_index > from_index:
            points_earned += 1

    # Criterion: There are no unrecognized words 
    recognized_words = ['SELECT', 'FROM', item.table_name.upper()] + [field.name.upper() for field in item.select]
    unrecognized_words = [token for token in tokens_upper if token not in recognized_words]
    total_points += 1
    if len(unrecognized_words) == 0:
        points_earned += 1
 
    return (points_earned, total_points)


def evaluate_cs1_prediction_score(item, predicted_sql_statement):
    (points_earned_part1, total_points_part1) = evaluate_cs1_prediction_score_part1(item, predicted_sql_statement)
    (points_earned_part2, total_points_part2) = evaluate_cs1_prediction_score_part2(item, predicted_sql_statement)

    return (points_earned_part1+points_earned_part2, total_points_part1+total_points_part2)


def evaluate_cs1_prediction(item: BatchItem, predicted_sql_statement: str) -> float:
    (points_earned_part, total_points_part) = evaluate_cs1_prediction_score(item, predicted_sql_statement)

    accuracy = 1.0 * (points_earned_part) / (total_points_part)

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
