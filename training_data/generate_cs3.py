from .fragments.models import BatchItem
from .generate_cs1 import evaluate_cs1_prediction_score
from .generate_cs2 import evaluate_cs2_prediction_score, generate_cs2


# Generate a batch of "command set 2" prompts and answers: SELECT xx FROM yy ORDER BY zz DESC
def generate_cs3(batch_size, order_by_clause_probability=0.9):
  return generate_cs2(batch_size, order_by_clause_probability, use_aggregates=True)


# Returns accuracy of a "command set 2" predicted answer compared to the ground truth
def evaluate_cs3_prediction_score(item: BatchItem, predicted_sql_statement: str):

    tokens = predicted_sql_statement.strip().split()
    tokens_upper = [token.upper().strip(',') for token in tokens]    

    recognized_words = ['SELECT', 'FROM', item.table_name.upper()]
    for field in item.select:
        field_sql_tokens = field.sql().upper().split()
        recognized_words.append(field_sql_tokens)
    print("recognized_words", recognized_words)

    unrecognized_words = [token for token in tokens_upper if token not in recognized_words]
    if len(unrecognized_words) == 0:
        return (1, 1)

    return (0,1)


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

    (cs1_points_earned, cs1_total_points) = evaluate_cs1_prediction_score(item, cs1_part)
    (cs2_points_earned, cs2_total_points) = evaluate_cs2_prediction_score(item, cs2_part)
    (cs3_points_earned, cs3_total_points) = evaluate_cs3_prediction_score(item, cs1_part)

    accuracy = 1.0 * (cs1_points_earned + cs2_points_earned + cs3_points_earned) / (cs1_total_points + cs2_total_points + cs3_total_points)
    
    return accuracy