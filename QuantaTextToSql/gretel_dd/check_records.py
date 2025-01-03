import re
import json
from datasets import load_dataset, Dataset, DatasetDict
import sqlglot
import sys
import pandas as pd  # Ensure pandas is imported

# 1. Append your custom modules path
sys.path.append("/mnt/foundation-shared/dhruv_gretel_ai/research/sql/quanta_text_to_sql")

# 2. Import custom validation functions
from QuantaTextToSql.training_data.generate_datasets import dict_to_batchitem
from QuantaTextToSql.training_data.generate_cs1 import evaluate_cs1_prediction

# 3. Load the dataset from Hugging Face
dataset_name = "withmartian/cs12_dataset"
dataset = load_dataset(dataset_name)

# 3.a. Add a unique 'id' to each record based on its index
def add_id(example, idx):
    return {'record_id': idx}

dataset = dataset.map(add_id, with_indices=True)

# 4. Define updated regex patterns
select_regex = re.compile(
    r'(?i)^SELECT\s+(\*|[a-zA-Z_][a-zA-Z0-9_]*(\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;\s*$'
)

create_table_regex = re.compile(
    r'(?i)^CREATE\s+TABLE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*\s+[A-Z]+(\s*\(\s*\d+\s*(,\s*\d+)?\s*\))?)(\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*\s+[A-Z]+(\s*\(\s*\d+\s*(,\s*\d+)?\s*\))?)*\s*\)\s*;\s*$'
)

# 5. Initialize counters and storage for invalid records
total_records = 0
valid_select = 0
valid_create = 0
valid_custom = 0
invalid_records = []
invalid_custom_records = []

# 6. Initialize schema dictionary
schema = {}

# 7. Define a threshold for custom validation (adjust as needed)
threshold = 0.99  # Example threshold

# 8. Iterate through each split in the dataset
for split in dataset.keys():
    print(f"Processing split: {split}")
    for record in dataset[split]:
        total_records += 1
        
        # Extract 'sql' and 'sql_context' fields
        sql = record.get('sql_statement', '').strip()
        sql_context = record.get('create_statement', '').strip()
        
        # Validate 'sql_context' (CREATE TABLE)
        if create_table_regex.match(sql_context):
            try:
                # Parse CREATE TABLE statement using sqlglot
                parsed_create = sqlglot.parse_one(sql_context)
                table_name = parsed_create.this.this  # Extract table name
                
                # Extract column names
                columns = parsed_create.expressions  # List of columns
                column_names = [col.this.this for col in columns]
                
                # Update schema
                schema[table_name] = column_names
                
                valid_create += 1
                
                # Now validate 'sql' (SELECT)
                if select_regex.match(sql):
                    try:
                        # Parse SELECT statement using sqlglot
                        parsed_select = sqlglot.parse_one(sql)
                        
                        # Check if 'from' exists
                        from_clause = parsed_select.args.get('from')
                        if from_clause and from_clause.expressions:
                            select_table = from_clause.expressions[0].this.name
                            
                            # Check if table exists in schema
                            if select_table in schema:
                                # Check if '*' is used
                                if any(expr.kind == 'STAR' for expr in parsed_select.expressions):
                                    # '*' implies all columns are selected; no need to check specific columns
                                    valid_select += 1
                                else:
                                    # Extract selected columns
                                    select_columns = [col.name for col in parsed_select.expressions]
                                    
                                    # Check if all selected columns exist in schema
                                    if all(col in schema[select_table] for col in select_columns):
                                        valid_select += 1
                                    else:
                                        invalid_records.append({
                                            'record_id': record.get('record_id', total_records),
                                            'sql': sql,
                                            'sql_context': sql_context,
                                            'error': 'Selected columns do not exist in CREATE TABLE schema.'
                                        })
                            else:
                                invalid_records.append({
                                    'record_id': record.get('record_id', total_records),
                                    'sql': sql,
                                    'sql_context': sql_context,
                                    'error': 'SELECT statement references a non-existent table.'
                                })
                        else:
                            invalid_records.append({
                                'record_id': record.get('record_id', total_records),
                                'sql': sql,
                                'sql_context': sql_context,
                                'error': 'SELECT statement missing FROM clause or has empty FROM clause.'
                            })
                    except sqlglot.errors.ParseError as e:
                        invalid_records.append({
                            'record_id': record.get('record_id', total_records),
                            'sql': sql,
                            'sql_context': sql_context,
                            'error': f'SQLGlot parse error in SELECT statement: {str(e)}'
                        })
                else:
                    invalid_records.append({
                        'record_id': record.get('record_id', total_records),
                        'sql': sql,
                        'sql_context': sql_context,
                        'error': 'Invalid SELECT statement syntax.'
                    })
                
                # Now apply your custom validator
                try:
                    # Convert the record to a batch item
                    item = dict_to_batchitem(record)
                    
                    # Prepare ground truth SQL without semicolon
                    gt_sql = sql.replace(";", "")
                    
                    # Evaluate the ground truth SQL
                    label_score = evaluate_cs1_prediction(item, gt_sql)
                    
                    # Determine if the custom validation passes based on scores
                    if label_score >= threshold:
                        valid_custom += 1
                    else:
                        #import ipdb; ipdb.set_trace()
                        invalid_custom_records.append({
                            'record_id': record.get('record_id', total_records),
                            'sql': sql,
                            'sql_context': sql_context,
                            'label_score': label_score,
                            'error': 'Custom validation failed based on gt scores.'
                        })
                except Exception as e:
                    invalid_custom_records.append({
                        'record_id': record.get('record_id', total_records),
                        'sql': sql,
                        'sql_context': sql_context,
                        'error': f'Custom validator error: {str(e)}'
                    })
                
            except sqlglot.errors.ParseError as e:
                invalid_records.append({
                    'record_id': record.get('record_id', total_records),
                    'sql': sql,
                    'sql_context': sql_context,
                    'error': f'SQLGlot parse error in CREATE TABLE statement: {str(e)}'
                })
        else:
            # Invalid CREATE TABLE statement
            invalid_records.append({
                'record_id': record.get('record_id', total_records),
                'sql': sql,
                'sql_context': sql_context,
                'error': 'Invalid CREATE TABLE statement syntax.'
            })

# 9. Summary of Validation
print("\n--- Validation Summary ---")
print(f"Total Records Processed: {total_records}")
print(f"Valid CREATE TABLE Statements: {valid_create}")
print(f"Valid SELECT Statements: {valid_select}")
print(f"Valid Custom Validations: {valid_custom}")
print(f"Invalid Records (Regex and SQLGlot): {len(invalid_records)}")
print(f"Invalid Records (Custom Validation): {len(invalid_custom_records)}")

# 10. Create Hugging Face Datasets
# Combine all invalid records
#combined_invalid = invalid_records + invalid_custom_records
combined_invalid = invalid_custom_records

# Check if combined_invalid is not empty
if combined_invalid:
    # Convert list of dicts to pandas DataFrame
    df_invalid = pd.DataFrame(combined_invalid)
    
    # Create Dataset for invalid records
    invalid_dataset = Dataset.from_pandas(df_invalid)
    
    # Upload the dataset to Hugging Face Hub
    invalid_dataset.push_to_hub("withmartian/cs12_invalid", private=True)
    
    print("Invalid records have been uploaded to Hugging Face as 'cs12_invalid'.")
else:
    print("No invalid records to upload.")

# 11. Filter out invalid records to create valid dataset
import ipdb; ipdb.set_trace()
original_dataset = dataset

# Extract invalid record IDs
invalid_ids = set(record['record_id'] for record in combined_invalid if 'record_id' in record)

def is_valid(record):
    return record.get('record_id') not in invalid_ids

# Apply filter to each split
valid_dataset_dict = DatasetDict()
for split in original_dataset.keys():
    print(f"Filtering split: {split}")
    split_dataset = original_dataset[split]
    filtered_dataset = split_dataset.filter(is_valid)
    valid_dataset_dict[split] = filtered_dataset

# 12. Upload valid records to Hugging Face as "cs11_valid"
valid_dataset_dict.push_to_hub("withmartian/cs12_valid", private=True)

print("Valid records have been uploaded to Hugging Face as 'cs12_valid'.")
