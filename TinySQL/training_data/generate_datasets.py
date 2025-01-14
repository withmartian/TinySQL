import sys
#sys.path.append('/mnt/foundation-shared/dhruv_gretel_ai/research/sql/quanta_text_to_sql')

import pandas as pd
import json
from datasets import Dataset, DatasetDict
from TinySQL.training_data.generate_cs1 import generate_cs1, evaluate_cs1_prediction
from TinySQL.training_data.generate_cs2 import generate_cs2, evaluate_cs2_prediction
from TinySQL.training_data.generate_cs3 import generate_cs3, evaluate_cs3_prediction
from TinySQL.training_data.fragments.models import TableName, BatchItem, TableField, SelectField, OrderField

def batchitem_to_dict(batch_item):
    d = {}
    d['command_set'] = batch_item.command_set
    
    d['table_name'] = batch_item.table_name.name
    d['table_name_synonym'] = batch_item.table_name.synonym
    d['table_name_use_synonym'] = batch_item.table_name.use_synonym

    d['create_statement'] = batch_item.create_statement
    d['english_prompt'] = batch_item.english_prompt
    d['sql_statement'] = batch_item.sql_statement

    # Convert 'table_fields' to JSON string
    d['table_fields'] = json.dumps([vars(tf) for tf in batch_item.table_fields])

    # Convert 'select' to JSON string
    d['select'] = json.dumps([vars(sf) for sf in batch_item.select])

    # 'order_by' may be None
    if batch_item.order_by is not None:
        d['order_by'] = json.dumps([vars(ob) for ob in batch_item.order_by])
    else:
        d['order_by'] = None

    return d

def dict_to_batchitem(row):
   # Reconstruct 'table_fields' from JSON string to list of TableField objects
    table_fields_list = json.loads(row['table_fields'])
    table_fields = [TableField(**tf_dict) for tf_dict in table_fields_list]

    # Reconstruct 'select' from JSON string to list of SelectField objects
    select_fields_list = json.loads(row['select'])
    select_fields = [SelectField(**sf_dict) for sf_dict in select_fields_list]

    # Reconstruct 'order_by' if it's not None
    if row['order_by'] is not None:
        order_by_list = json.loads(row['order_by'])
        order_by_fields = [OrderField(**ob_dict) for ob_dict in order_by_list]
    else:
        order_by_fields = None

    # Create a new BatchItem instance
    batch_item = BatchItem(
        command_set=row['command_set'],
        table_name=TableName(name=row['table_name'], synonym=row['table_name_synonym'], use_synonym=row['table_name_use_synonym']), 
        table_fields=table_fields,
        create_statement=row['create_statement'],
        select=select_fields,
        order_by=order_by_fields,
        english_prompt=row['english_prompt'],
        sql_statement=row['sql_statement']
    )

    return batch_item

def generate_dataset(batch_size, generate_cs_function, evaluate_cs_function, dataset_name, use_synonyms : bool, push_to_hf : bool):
    # Generate dataset using the provided CS function
    dataset = generate_cs_function(batch_size, use_synonyms=use_synonyms)

    # Create a dataframe
    dataset_dicts = [batchitem_to_dict(item) for item in dataset]

    # Convert to HF dataset
    df = pd.DataFrame(dataset_dicts)
    hf_dataset = Dataset.from_pandas(df)

    # Split dataset into train + val (90%) and test (10%)
    train_val_split = hf_dataset.train_test_split(test_size=0.1, seed=420)
    test_dataset = train_val_split['test']

    # Further split train_val into train (85%) and val (15% of the remaining data)
    train_val_split = train_val_split['train'].train_test_split(test_size=0.15, seed=420)

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
    if push_to_hf:
        hf_dataset.push_to_hub(dataset_name)

    # Unit tests for the dataset (on the train set as an example)
    accuracy = 0
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        item = dict_to_batchitem(sample)
        score = evaluate_cs_function(item, sample['sql_statement'])
        if (not push_to_hf) or score < 1:
            print("Table:", sample['table_name'])
            print("Table fields:", sample['table_fields'])
            print("Create:", sample['create_statement'])
            print("Selected fields:", sample['table_fields'])
            print("English:", sample['english_prompt'])
            print("SQL:", sample['sql_statement'])
        accuracy += score
    print(f"Dataset '{dataset_name}' Accuracy:", (accuracy / len(train_dataset)) * 100)

# Main function
if __name__ == '__main__':
    batch_size = 100000
    use_synonyms = True
    suffix = "_synonyms" if use_synonyms else ""

    generate_dataset(batch_size, generate_cs1, evaluate_cs1_prediction, "withmartian/cs1_dataset" + suffix, use_synonyms, True)
    generate_dataset(batch_size, generate_cs2, evaluate_cs2_prediction, "withmartian/cs2_dataset" + suffix, use_synonyms, True)
    generate_dataset(batch_size, generate_cs3, evaluate_cs3_prediction, "withmartian/cs3_dataset" + suffix, use_synonyms, True)
