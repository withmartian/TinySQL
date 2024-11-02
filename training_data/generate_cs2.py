from .sql_create_table import get_sql_create_table
from .sql_select_from import get_sql_select_from
from .sql_order_by import get_sql_order_by
from .fragments.models import BatchItem
from .generate_cs1 import evaluate_cs1_prediction_score, get_english_select_from, get_english_order_by


# Generate a batch of "command set 2" prompts and answers: SELECT xx FROM yy ORDER BY zz DESC
def generate_cs2(batch_size, order_by_clause_probability=0.9, use_aggregates=False):
  

    batch = []
    for i in range(batch_size):
        (table_name, table_fields, create_table_statement) = get_sql_create_table(2, 12)

        (selected_fields, sql_select_statement) = get_sql_select_from(table_name, table_fields, use_aggregates)
        english_select_from_prompt = get_english_select_from(table_name, selected_fields)

        # Randomly decide whether to include an ORDER BY clause
        include_order_by = i < batch_size * order_by_clause_probability
        if include_order_by:
            (order_by_fields, sql_order_by_statement) = get_sql_order_by(table_fields)
            english_order_by_prompt = get_english_order_by(order_by_fields)
        else:
            order_by_fields = []
            english_order_by_prompt = ""
            sql_order_by_statement = ""

        batch_item = BatchItem(
            command_set=2,
            table_name=table_name,
            table_fields=table_fields,
            create_statement=create_table_statement,
            select=selected_fields,
            order_by=order_by_fields,
            english_prompt=english_select_from_prompt + " " + english_order_by_prompt,
            sql_statement=sql_select_statement + " " + sql_order_by_statement, # ground truth
        )

        batch.append(batch_item)

    return batch

# Returns accuracy of a "command set 2" predicted answer compared to the ground truth
def evaluate_cs2_prediction_score(item: BatchItem, predicted_sql_statement: str):

    # Handle the no ORDER BY clause case
    if predicted_sql_statement == "":
        return (0,0)

    # We want to reward components of the answer that are correct.
    # - Starts with ORDER BY                1 point
    # - Contains field names                N points
    # - There are no unrecognised words     1 point

    total_points = 0
    points_earned = 0

    # Tokenize the predicted SQL statement
    tokens = predicted_sql_statement.strip().split()
    tokens_upper = [token.upper().strip(',') for token in tokens]

    # Criterion: Starts with ORDER BY
    total_points += 1
    if (tokens_upper[0] == 'ORDER' and tokens_upper[1] == 'BY'):
        points_earned += 1

    # Criterion: Contains field names
    for field in item.order_by:
        total_points += 1
        if field.name.upper() in tokens_upper:
            points_earned += 1

    # Criterion: There are no unrecognized words 
    recognized_words = ['ORDER', 'BY', 'ASC', 'DESC'] + [field.name.upper() for field in item.order_by]
    unrecognized_words = [token for token in tokens_upper if token not in recognized_words]
    total_points += 1
    if len(unrecognized_words) == 0:
        points_earned += 1

    return (points_earned, total_points)


def evaluate_cs2_prediction(item: BatchItem, predicted_sql_statement: str) -> float:

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

    accuracy = 1.0 * (cs1_points_earned + cs2_points_earned) / (cs1_total_points + cs2_total_points)
    
    return accuracy
