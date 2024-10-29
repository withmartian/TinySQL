from training_data import get_sql_table_names, get_sql_field_names, get_sql_create_table, get_sql_select_from, get_english_select_from_phrase
from .batch_item import BatchItem


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
            command_set=1, 
            table_name=table_name,
            table_fields=table_fields,
            create_statement=create_table_statement,
            selected_fields=selected_fields,
            order_by_fields=None,
            english_prompt=english_select_from_prompt,
            sql_statement=sql_select_statement, # ground truth
        )

        batch.append(batch_item)

    return batch


# Returns accuracy of a "command set 1" predicted answer compared to the ground truth
def evaluate_cs1_prediction_score(item: BatchItem, predicted_sql_statement: str):

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
    N = len(item.selected_fields)
    total_points = 7 + 2 * N  


    points_earned = 0

    # Tokenize the predicted SQL statement
    tokens = predicted_sql_statement.strip().split()
    tokens_upper = [token.upper().strip(',') for token in tokens]

    # Criterion 1: Contains SELECT (1 point)
    if 'SELECT' in tokens_upper:
        points_earned += 1

    # Criterion 2: Contains table_name (1 point)
    if item.table_name.upper() in tokens_upper:
        points_earned += 1

    # Criterion 3: Contains FROM (1 point)
    if 'FROM' in tokens_upper:
        points_earned += 1

    # Criterion 4: Contains field names (N points)
    for field in item.selected_fields:
        if field.upper() in tokens_upper:
            points_earned += 1

    # Criterion 5: First word is SELECT (1 point)
    if len(tokens_upper) > 0 and tokens_upper[0] == 'SELECT':
        points_earned += 1

    # Criterion 6: All field names are after SELECT (N points)
    if 'SELECT' in tokens_upper:
        select_index = tokens_upper.index('SELECT')
        for field in item.selected_fields:
            field_indices = [i for i, token in enumerate(tokens_upper) if token == field.upper()]
            if all(i > select_index for i in field_indices):
                points_earned += 1

    # Criterion 7: Word FROM is after all field names (1 point)
    if 'FROM' in tokens_upper:
        from_index = tokens_upper.index('FROM')
        last_field_index = max(
            [i for i, token in enumerate(tokens_upper) if token in [f.upper() for f in item.selected_fields]],
            default=-1
        )
        if from_index > last_field_index:
            points_earned += 1

    # Criterion 8: table_name is after FROM (1 point)
    if 'FROM' in tokens_upper and item.table_name.upper() in tokens_upper:
        from_index = tokens_upper.index('FROM')
        table_name_index = tokens_upper.index(item.table_name.upper())
        if table_name_index > from_index:
            points_earned += 1

    # Criterion 9: There are no unrecognized words (1 point)
    recognized_words = ['SELECT', 'FROM'] + [item.table_name.upper()] + [field.upper() for field in item.selected_fields]
    unrecognized_words = [token for token in tokens_upper if token not in recognized_words]
    if len(unrecognized_words) == 0:
        points_earned += 1

    return (points_earned, total_points)


def evaluate_cs1_prediction(item: BatchItem, predicted_sql_statement: str) -> float:
    (points_earned, total_points) = evaluate_cs1_prediction_score(item, predicted_sql_statement)

    accuracy = 1.0 * points_earned / total_points

    return accuracy