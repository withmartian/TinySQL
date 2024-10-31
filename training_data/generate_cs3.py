from .fragments.models import BatchItem
from .generate_cs1 import evaluate_cs1_prediction_score_part1
from .generate_cs2 import evaluate_cs2_prediction_score, generate_cs2


# Generate a batch of "command set 3" prompts and answers: SELECT MAX(x1), MIN(x2), x3, SUM(x4) FROM yy ORDER BY zz DESC
def generate_cs3(batch_size, order_by_clause_probability=0.9):
  return generate_cs2(batch_size, order_by_clause_probability, use_aggregates=True)


# Returns accuracy of a "command set 3" predicted answer compared to the ground truth
def evaluate_cs3_prediction_score(item: BatchItem, predicted_sql_statement: str):

    # We want to reward components of the answer that are correct.
    # - Contains selected field names with aggregates       N points    
    # - Contains "as" field names for aggregated fields     N points    
    # - All field names are after SELECT                    N points
    # - Word FROM is after all field names                  1 point
    # - Word table_name is after FROM                       1 point   
    # - There are no unrecognized words                     1 point

    tokens = predicted_sql_statement.strip().split()
    tokens_upper = [token.upper().strip(',') for token in tokens]    

    total_points = 0  
    points_earned = 0

    # Criterion: Contains selected field names with (optional) aggregates
    for field in item.select:
        total_points += 1
        if field.aggregate_of_field.upper() in tokens_upper:
            points_earned += 1

   # Criterion: Contains "as" field names for aggregated fields 
    for field in item.select:
        if field.aggregate != "":
            total_points += 1
            if field.aggregated_name.upper() in tokens_upper:
                points_earned += 1

    # Criterion: All field names are after SELECT  
    if 'SELECT' in tokens_upper:
        select_index = tokens_upper.index('SELECT')
        for field in item.select:
            field_indices = [i for i, token in enumerate(tokens_upper) if token == field.aggregate_of_field.upper()]

            total_points += 1
            if all(i > select_index for i in field_indices):
                points_earned += 1

    # Criterion: Word FROM is after all field names 
    if 'FROM' in tokens_upper:
        from_index = tokens_upper.index('FROM')
        last_field_index = max(
            [i for i, token in enumerate(tokens_upper) if token in [f.aggregate_of_field.upper() for f in item.select]],
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
    recognized_words = ['SELECT', 'AS', 'FROM', item.table_name.upper()]
    recognized_words += [field.aggregate_of_field.upper() for field in item.select]
    recognized_words += [field.aggregated_name.upper() for field in item.select if field.aggregate != ""]
    unrecognized_words = [token for token in tokens_upper if token not in recognized_words]
    total_points += 1
    if len(unrecognized_words) == 0:
        points_earned += 1
 
    return (points_earned, total_points)


def evaluate_cs3_prediction(item: BatchItem, predicted_sql_statement: str) -> float:

    # Separate predicted_sql_statement into portion before ORDER BY and portion including and after ORDER BY
    split_phrase = "ORDER BY"
    predicted_sql_statement = predicted_sql_statement.strip()   
    predicted_sql_parts = predicted_sql_statement.split(split_phrase)

    cs1_part = predicted_sql_parts[0]
    if len(predicted_sql_parts) == 1:
        cs2_part = ""
    else:
        cs2_part = split_phrase + " " + predicted_sql_parts[1]

    (cs1_points_earned, cs1_total_points) = evaluate_cs1_prediction_score_part1(item, cs1_part)
    (cs2_points_earned, cs2_total_points) = evaluate_cs2_prediction_score(item, cs2_part)
    (cs3_points_earned, cs3_total_points) = evaluate_cs3_prediction_score(item, cs1_part)

    accuracy = 1.0 * (cs1_points_earned + cs2_points_earned + cs3_points_earned) / (cs1_total_points + cs2_total_points + cs3_total_points)
    
    return accuracy