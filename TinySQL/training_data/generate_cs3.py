from .fragments.models import BatchItem
from .generate_cs1 import evaluate_cs1_prediction_score_part1, trim_newlines_and_multiple_spaces, evaluate_unrecognised_words
from .generate_cs2 import evaluate_cs2_prediction_score, generate_cs2


# Generate a batch of "command set 3" prompts and answers: SELECT MAX(x1), MIN(x2), x3, SUM(x4) FROM yy ORDER BY zz DESC
def generate_cs3(batch_size, order_by_clause_probability=0.9, min_cols=2, max_cols=12):
  return generate_cs2(batch_size, order_by_clause_probability, use_aggregates=True, min_cols=min_cols, max_cols=max_cols)


# Returns accuracy of a "command set 3" predicted answer compared to the ground truth
def evaluate_cs3_prediction_score(item: BatchItem, predicted_sql_statement: str):

    # We want to reward components of the answer that are correct.
    # - Contains selected field names with aggregates       N points    
    # - Contains "as" field names for aggregated fields     N points    
    # - All field names are after SELECT                    N points
    # - Word FROM is after all field names                  1 point
    # - Word table_name is after FROM                       1 point   
    # - There are no unrecognized words                     1 point

    tokens = predicted_sql_statement.split()
    test_tokens = [token.strip(',') for token in tokens]    

    total_points = 0  
    points_earned = 0

    # Criterion: Contains selected field names with (optional) aggregates
    for field in item.select:
        total_points += 1
        if field.aggregate_of_field.upper() in test_tokens:
            points_earned += 1

   # Criterion: Contains "as" field names for aggregated fields 
    for field in item.select:
        if field.aggregate != "":
            total_points += 1
            if field.aggregated_name.upper() in test_tokens:
                points_earned += 1

    # Criterion: All field names are after SELECT  
    if 'SELECT' in test_tokens:
        select_index = test_tokens.index('SELECT')
        for field in item.select:
            field_indices = [i for i, token in enumerate(test_tokens) if token == field.aggregate_of_field.upper()]

            total_points += 1
            if all(i > select_index for i in field_indices):
                points_earned += 1

    # Criterion: Word FROM is after all field names 
    if 'FROM' in test_tokens:
        from_index = test_tokens.index('FROM')
        last_field_index = max(
            [i for i, token in enumerate(test_tokens) if token in [f.aggregate_of_field.upper() for f in item.select]],
            default=-1
        )
        total_points += 1
        if from_index > last_field_index:
            points_earned += 1

    # Criterion: table_name is after FROM 
    if 'FROM' in test_tokens and item.table_name.upper() in test_tokens:
        from_index = test_tokens.index('FROM')
        table_name_index = test_tokens.index(item.table_name.upper())

        total_points += 1
        if table_name_index > from_index:
            points_earned += 1

    # Criterion: There are no unrecognized words 
    recognized_words = ['SELECT', 'AS', 'FROM', item.table_name.upper()]
    recognized_words += [field.aggregate_of_field.upper() for field in item.select]
    recognized_words += [field.aggregated_name.upper() for field in item.select if field.aggregate != ""]
    (earned, possible) = evaluate_unrecognised_words(recognized_words, test_tokens)
    total_points += possible
    points_earned += earned
 
    return (points_earned, total_points)


def evaluate_cs3_prediction(item: BatchItem, predicted_sql_statement: str) -> float:

    test_sql_statement = trim_newlines_and_multiple_spaces(predicted_sql_statement).upper()
    if test_sql_statement == "":
        return 0.0
    
    # Separate predicted_sql_statement into portion before ORDER BY and portion including and after ORDER BY
    split_phrase = "ORDER BY"
    test_sql_parts = test_sql_statement.split(split_phrase)

    cs1_part = test_sql_parts[0]
    if len(test_sql_parts) == 1:
        cs2_part = ""
    else:
        cs2_part = split_phrase + " " + test_sql_parts[1]

    (cs1_points_earned, cs1_total_points) = evaluate_cs1_prediction_score_part1(item, cs1_part)
    (cs2_points_earned, cs2_total_points) = evaluate_cs2_prediction_score(item, cs2_part)
    (cs3_points_earned, cs3_total_points) = evaluate_cs3_prediction_score(item, cs1_part)

    accuracy = 1.0 * (cs1_points_earned + cs2_points_earned + cs3_points_earned) / (cs1_total_points + cs2_total_points + cs3_total_points)
    
    return accuracy

def evaluate_cs3_predictions(items, predicted_sql_statements) -> float:
    total_accuracy = 0.0
    for i, item in enumerate(items):
        accuracy = evaluate_cs3_prediction(item, predicted_sql_statements[i])
        total_accuracy += accuracy
    return total_accuracy / len(items)  