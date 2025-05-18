import random
from .sql_create_table import get_sql_create_table
from .sql_select_from import get_sql_select_from
from .sql_order_by import get_sql_order_by
from .sql_where import get_sql_where
from .fragments.models import BatchItem, TableName
from .fragments.english_where import get_english_where
from .generate_cs1 import get_english_select_from, trim_newlines_and_multiple_spaces, evaluate_unrecognised_words
from .generate_cs2 import get_english_order_by


def generate_cs4(batch_size, order_by_clause_probability=0.9, where_clause_probability=0.8,
                 use_aggregates=False, min_cols=2, max_cols=12,
                 use_synonyms_table=False, use_synonyms_field=False):
    batch = []
    for i in range(batch_size):
        (table_name, table_fields, create_table_statement) = get_sql_create_table(min_cols=min_cols, max_cols=max_cols)
        (selected_fields, sql_select_statement) = get_sql_select_from(table_name, table_fields, use_aggregates)
        (english_select_from_prompt, table_name, selected_fields), agg_phrases = get_english_select_from(
            table_name, selected_fields, use_synonyms_table, use_synonyms_field
        )

        include_where = random.random() < where_clause_probability
        if include_where:
            _, _, conditions, sql_where_statement = get_sql_where(table_fields)
            english_where_prompt = get_english_where(conditions)
        else:
            conditions = []
            sql_where_statement = ""
            english_where_prompt = ""

        include_order_by = i < batch_size * order_by_clause_probability
        if include_order_by:
            (order_by_fields, sql_order_by_statement) = get_sql_order_by(table_fields)
            english_order_by_prompt, order_by_phrase = get_english_order_by(order_by_fields)
        else:
            order_by_fields = []
            sql_order_by_statement = ""
            english_order_by_prompt = ""
            order_by_phrase = ""
        full_sql_statement = (sql_select_statement + " " + sql_where_statement + " " + sql_order_by_statement).strip()
        full_english_prompt = (english_select_from_prompt + " " + english_where_prompt + " " + english_order_by_prompt).strip()
        batch_item = BatchItem(
            command_set=4,
            table_name=TableName(name=table_name.name, synonym=table_name.synonym, use_synonym=table_name.use_synonym),
            table_fields=table_fields,
            create_statement=create_table_statement,
            select=selected_fields,
            order_by=order_by_fields,
            english_prompt=full_english_prompt,
            sql_statement=full_sql_statement,
            order_by_phrase=order_by_phrase,
            agg_phrases=agg_phrases,
            where=conditions
        )
        batch.append(batch_item)
    return batch


# placeholder -- deprioritized for rebuttal
def evaluate_cs4_prediction_score(item: BatchItem, predicted_sql_statement: str):
    return (0, 0)


#placeholder -- deprioritized for rebuttal
def evaluate_cs4_prediction(item: BatchItem, predicted_sql_statement: str) -> float:
    return 0


# placeholder -- deprioritized for rebuttal
def evaluate_cs4_predictions(items, predicted_sql_statements) -> float:
    total_accuracy = 0.0
    for i, item in enumerate(items):
        accuracy = evaluate_cs4_prediction(item, predicted_sql_statements[i])
        total_accuracy += accuracy
    return total_accuracy / len(items)
