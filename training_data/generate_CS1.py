import random
from .fragments.english_aggregates import get_english_sum_phrases, get_english_avg_phrases, get_english_min_phrases, get_english_max_phrases, get_english_count_phrases
from .fragments.english_select_from import get_english_select_from_phrase
from .fragments.english_order_by import get_english_order_by_phrase
from .sql_create_table import get_sql_create_table
from .sql_select_from import get_sql_select_from
from .fragments.models import BatchItem, OrderField, SelectField


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


    # Calculate the total number of points on offer
    N = len(item.select)
    total_points = 6 + 2 * N  


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
    for field in item.select:
        if field.name.upper() in tokens_upper:
            points_earned += 1

    # Criterion 5: First word is SELECT (1 point)
    if len(tokens_upper) > 0 and tokens_upper[0] == 'SELECT':
        points_earned += 1

    # Criterion 6: All field names are after SELECT (N points)
    if 'SELECT' in tokens_upper:
        select_index = tokens_upper.index('SELECT')
        for field in item.select:
            field_indices = [i for i, token in enumerate(tokens_upper) if token == field.name.upper()]
            if all(i > select_index for i in field_indices):
                points_earned += 1

    # Criterion 7: Word FROM is after all field names (1 point)
    if 'FROM' in tokens_upper:
        from_index = tokens_upper.index('FROM')
        last_field_index = max(
            [i for i, token in enumerate(tokens_upper) if token in [f.name.upper() for f in item.select]],
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

    return (points_earned, total_points)


def evaluate_cs1_prediction(item: BatchItem, predicted_sql_statement: str) -> float:
    (points_earned, total_points) = evaluate_cs1_prediction_score(item, predicted_sql_statement)

    tokens = predicted_sql_statement.strip().split()
    tokens_upper = [token.upper().strip(',') for token in tokens]    

    recognized_words = ['SELECT', 'FROM', item.table_name.upper()] + [field.name.upper() for field in item.select]

    unrecognized_words = [token for token in tokens_upper if token not in recognized_words]
    if len(unrecognized_words) == 0:
        points_earned += 1
    total_points += 1

    accuracy = 1.0 * points_earned / total_points

    return accuracy