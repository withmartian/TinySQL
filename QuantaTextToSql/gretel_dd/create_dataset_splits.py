import re
import json
from datasets import load_dataset, Dataset, DatasetDict

# Load the dataset from Hugging Face
dataset_name = "withmartian/cs12_dataset"
dataset = load_dataset(dataset_name)

# Find records with an empty create_statement
def find_empty_create_statement(example):
    return example["create_statement"] == ""

def has_non_empty_create_statement(example):
    return example["create_statement"].strip() != ""

# Filter the dataset for empty create_statement records
empty_records = dataset["train"].filter(find_empty_create_statement)

# Display the count and examples of empty records
print(f"Number of records with empty create_statement: {len(empty_records)}")
if len(empty_records) > 0:
    print("Examples of records with empty create_statement:")
    for record in empty_records.select(range(min(5, len(empty_records)))):
        print(record)

#import ipdb; ipdb.set_trace()
dataset = dataset.filter(has_non_empty_create_statement)

# Rename columns
def rename_columns(example):
    example["english_prompt"] = example.pop("sql_prompt")
    example["create_statement"] = example.pop("sql_context")
    example["sql_statement"] = example.pop("sql")
    return example

# dataset = dataset.map(rename_columns)

# Define table creation function
def create_table_field(create_statement):
    table_fields = []
    try:
        # Extract the field definitions between parentheses
        match = re.search(r"\((.+)\)", create_statement, re.DOTALL)
        if not match:
            #import ipdb; ipdb.set_trace()
            print(f"Invalid create_statement: {create_statement}")
            return []
        
        # Extract fields while respecting nested parentheses
        fields = re.split(r",(?![^(]*\))", match.group(1))  # Split on commas not within parentheses
        for field in fields:
            # Strip and split into name and type
            field = field.strip()
            parts = field.split(maxsplit=1)
            if len(parts) == 2:
                name, field_type = parts
                table_fields.append({"name": name, "type": field_type})
            else:
                print(f"Skipping invalid field: {field} in create_statement: {create_statement}")
    except Exception as e:
        import ipdb; ipdb.set_trace()
        print(f"Error processing create_statement: {create_statement} - {e}")
    return table_fields

# Iterate over the dataset to generate table fields
def add_table_fields(example):
    try:
        table_fields_list = create_table_field(example["create_statement"])
        
        # Convert the list to a string
        example["table_fields"] = json.dumps(table_fields_list)  # Convert to JSON-formatted string
    except Exception as e:
        print(f"Error adding table fields for example: {example} - {e}")
        example["table_fields"] = "[]"  # Default to an empty list for problematic cases
    return example

# Apply the function to the dataset
#import ipdb; ipdb.set_trace()
#dataset = dataset.map(add_table_fields)

# Add a new column "order_by" initialized to None (null in serialization)
def add_order_by_column(example):
    example["order_by"] = None  # Initialize with None
    return example

# Apply the function to add the column
#import ipdb; ipdb.set_trace()
#dataset = dataset.map(add_order_by_column)

# Add a new column "command_set" initialized to "1-2"
def add_command_set_column(example):
    example["command_set"] = 1 #"1-2"  #TODO: CHANGE THIS
    return example

# Apply the function to add the column
#import ipdb; ipdb.set_trace()
#dataset = dataset.map(add_command_set_column)

def add_selected_fields(example):
    try:
        #import ipdb; ipdb.set_trace()
        #TODO: CHANGE THIS
        selected_fields = example["selected_columns"].split(",")
        #selected_fields = example["column_name"].split(",")
        selected_fields = [field.strip() for field in selected_fields]  # Remove extra spaces
        select_list = [{"name": field, "aggregate": ""} for field in selected_fields]

        example["select"] = json.dumps(select_list)  # Convert to JSON-formatted string
    except Exception as e:
        print(f"Error processing selected_columns for example: {example} - {e}")
        example["select"] = "[]"  # Default to an empty JSON string for problematic cases
    return example

#import ipdb; ipdb.set_trace()
dataset = dataset.map(add_selected_fields)

# # Check the original train size
# original_train_size = len(dataset["train"])
# print(f"Original train dataset size: {original_train_size}")

# # Define the split proportions
# train_split = 0.8  # 80% for training
# val_split = 0.1    # 10% for validation
# test_split = 0.1   # 10% for testing

# # Perform the split
# data_split = dataset["train"].train_test_split(test_size=val_split + test_split, seed=42)

# # Split the remaining portion (val + test) into validation and test
# val_test_split = data_split["test"].train_test_split(test_size=test_split / (val_split + test_split), seed=42)

# # Combine splits into a DatasetDict
# final_splits = DatasetDict({
#     "train": data_split["train"],
#     "validation": val_test_split["train"],
#     "test": val_test_split["test"]
# })

# # Check the sizes of the splits
# print(f"Train dataset size: {len(final_splits['train'])}")
# print(f"Validation dataset size: {len(final_splits['validation'])}")
# print(f"Test dataset size: {len(final_splits['test'])}")

# dataset = Dataset.from_pandas(final_splits)
# final_splits.push_to_hub(dataset_name)
import ipdb; ipdb.set_trace()
#dataset.push_to_hub(dataset_name)

dataset.push_to_hub('test')
