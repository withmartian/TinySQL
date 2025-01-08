import random
from .fragments.english_aggregates import get_english_sum_phrases, get_english_avg_phrases, get_english_min_phrases, get_english_max_phrases, get_english_count_phrases
from .fragments.english_select_from import get_english_select_from_phrase
from .fragments.english_order_by import get_english_order_by_phrase
from .sql_create_table import get_sql_create_table
from .sql_select_from import get_sql_select_from
from .fragments.models import TableName, BatchItem, OrderField, SelectField, trim_newlines_and_multiple_spaces


def get_english_order_by(fields: list[OrderField]) -> str:
    answer = ""

    for i, field in enumerate(fields):
        english = get_english_order_by_phrase(field.asc)
        if i > 0:
            answer += ","
        answer += " " + english + " " + field.name
    
    return answer


def get_english_select_from(table_name: TableName, fields: list[SelectField], use_synonyms: bool) -> str:
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

        field_str = field.name # field.synonym if use_synonyms else field.name
        if phrases is None:
            english_field = field_str
        else:
            english_field = f"{random.choice(phrases)} {field_str}"

        english_fields += english_field
        
        if i == len(fields) - 2:
            english_fields += " and "   
        elif i < len(fields) - 2:
            english_fields += ", "   
       
    table_str = table_name.synonym if use_synonyms else table_name.name

    # Create English phrase
    english = template.replace("[fields]", english_fields).replace("[table]", table_str)
    
    return english


# Generate a batch of "command set 1" prompts and answers. These are SQL SELECT statements with a single table and a few fields.
def generate_cs1(batch_size, min_cols=2, max_cols=12, use_synonyms=False) -> list[BatchItem]:

    batch = []
    for i in range(batch_size):
        (table_name, table_fields, create_table_statement) = get_sql_create_table(min_cols, max_cols)

        (selected_fields, sql_select_statement) = get_sql_select_from(table_name, table_fields, False)

        english_select_from_prompt = get_english_select_from(table_name, selected_fields, use_synonyms)

        batch_item = BatchItem(
            command_set=1, 
            table_name=TableName(name=table_name.name, synonym=table_name.synonym),
            table_fields=table_fields,
            create_statement=create_table_statement,
            select=selected_fields,
            order_by=None,
            english_prompt=english_select_from_prompt,
            sql_statement=sql_select_statement, # ground truth
        )
        
        batch.append(batch_item)

    return batch


# Reduce score if the answer contains words we do not recognize
def evaluate_unrecognised_words(recognized_words, test_tokens):
    unrecognized_words = [token for token in test_tokens if token not in recognized_words]
    total_points = 10
    if len(unrecognized_words) == 0:
        points_earned = 10
    elif len(unrecognized_words) == 1:     
        points_earned = 8
    elif len(unrecognized_words) == 2:     
        points_earned = 6
    elif len(unrecognized_words) <= 4:     
        points_earned = 4
    elif len(unrecognized_words) <= 8:     
        points_earned = 2
    else:     
        points_earned = 0

    return (points_earned, total_points)


# Reduce score if the answer is much longer than the ground-truth answer. This covers duplicated text and verbose answers. 
def evaluate_answer_length(item: BatchItem, predicted_sql_statement: str):
    good_length = len(item.sql_statement)
    test_length = len(predicted_sql_statement)
    verbose_chars = max(0, test_length - good_length)

    total_points = 10
    if verbose_chars <= 2:
        points_earned = 10
    elif verbose_chars <= 4:     
        points_earned = 8
    elif verbose_chars <= 6:     
        points_earned = 6
    elif verbose_chars <= 8:     
        points_earned = 4
    elif verbose_chars <= 10:     
        points_earned = 2
    else:     
        points_earned = 0

    return (points_earned, total_points)


# Returns accuracy of a "command set 1" predicted answer compared to the ground truth
def evaluate_cs1_prediction_score_part1(item: BatchItem, predicted_sql_statement: str):

    # We want to reward components of the answer that are correct.
    # - Contains SELECT                     1 point
    # - Contains table_name                 1 point
    # - Contains FROM                       1 point
    # - First word is SELECT                1 point
    # - Answer length is reasonable         10 points

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
    if item.table_name.name.upper() in tokens_upper:
        points_earned += 1

    # Criterion: Contains FROM (1 point)
    total_points += 1
    if 'FROM' in tokens_upper:
        points_earned += 1

    # Criterion: First word is SELECT (1 point)
    total_points += 1
    if len(tokens_upper) > 0 and tokens_upper[0] == 'SELECT':
        points_earned += 1

    # Criterion: Answer length is reasonable
    (earned, possible) = evaluate_answer_length(item, predicted_sql_statement)
    total_points += possible
    points_earned += earned
    
    return (points_earned, total_points)


def evaluate_cs1_prediction_score_part2(item: BatchItem, predicted_sql_statement: str):

    # We want to reward components of the answer that are correct.
    # - Contains selected field names       N points    
    # - All field names are after SELECT    N points
    # - Word FROM is after all field names  1 point
    # - Word table_name is after FROM       1 point   
    # - There are no unrecognized words     10 point

    total_points = 0  
    points_earned = 0

    # Tokenize the predicted SQL statement
    tokens = predicted_sql_statement.split()
    test_tokens = [token.strip(',') for token in tokens]

    # Criterion: Contains field names
    for field in item.select:
        total_points += 1
        if field.name.upper() in test_tokens:
            points_earned += 1

    # Criterion: All field names are after SELECT 
    if 'SELECT' in test_tokens:
        select_index = test_tokens.index('SELECT')
        for field in item.select:
            field_indices = [i for i, token in enumerate(test_tokens) if token == field.name.upper()]

            total_points += 1
            if all(i > select_index for i in field_indices):
                points_earned += 1

    # Criterion: Word FROM is after all field names 
    if 'FROM' in test_tokens:
        from_index = test_tokens.index('FROM')
        last_field_index = max(
            [i for i, token in enumerate(test_tokens) if token in [f.name.upper() for f in item.select]],
            default=-1
        )
        total_points += 1
        if from_index > last_field_index:
            points_earned += 1

    # Criterion: table_name is after FROM 
    table_str = item.table_name.name.upper()
    if 'FROM' in test_tokens and table_str in test_tokens:
        from_index = test_tokens.index('FROM')
        table_name_index = test_tokens.index(table_str)

        total_points += 1
        if table_name_index > from_index:
            points_earned += 1

    # Criterion: There are no unrecognized words 
    recognized_words = ['SELECT', 'FROM', table_str] + [field.name.upper() for field in item.select]
    (earned, possible) = evaluate_unrecognised_words(recognized_words, test_tokens)
    total_points += possible
    points_earned += earned

    return (points_earned, total_points)


def evaluate_cs1_prediction_score(item: BatchItem, predicted_sql_statement):

    (points_earned_part1, total_points_part1) = evaluate_cs1_prediction_score_part1(item, predicted_sql_statement)
    (points_earned_part2, total_points_part2) = evaluate_cs1_prediction_score_part2(item, predicted_sql_statement)

    return (points_earned_part1+points_earned_part2, total_points_part1+total_points_part2)


def evaluate_cs1_prediction(item: BatchItem, predicted_sql_statement: str) -> float:

    test_sql_statement = trim_newlines_and_multiple_spaces(predicted_sql_statement).upper()
    if test_sql_statement == "":
        return 0.0

    (points_earned_part, total_points_part) = evaluate_cs1_prediction_score(item, test_sql_statement)

    accuracy = 1.0 * (points_earned_part) / (total_points_part)

    return accuracy

def evaluate_cs1_predictions(items, predicted_sql_statements) -> float:
    total_accuracy = 0.0
    for i, item in enumerate(items):
        accuracy = evaluate_cs1_prediction(item, predicted_sql_statements[i])
        total_accuracy += accuracy
    return total_accuracy / len(items)  
