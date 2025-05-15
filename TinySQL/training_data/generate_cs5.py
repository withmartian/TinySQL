import random
from .sql_create_table import get_sql_create_table
from .sql_select_from import get_sql_select_from
from .sql_order_by import get_sql_order_by
from .sql_where import get_sql_where
from .sql_join import get_sql_join
from .fragments.models import BatchItem, TableName
from .fragments.english_where import get_english_where
from .generate_cs1 import get_english_select_from, trim_newlines_and_multiple_spaces, evaluate_unrecognised_words
from .generate_cs2 import get_english_order_by


def generate_cs5(batch_size,
                 order_by_clause_probability=0.9,
                 where_clause_probability=0.8,
                 use_aggregates=False,
                 min_cols=2,
                 max_cols=12,
                 use_synonyms_table=False,
                 use_synonyms_field=False):
    batch = []
    for _ in range(batch_size):
        (main_table, main_fields, main_create_statement) = get_sql_create_table(
            min_cols=min_cols, max_cols=max_cols
        )
        
        (selected_fields, sql_select_statement) = get_sql_select_from(
            main_table, main_fields, use_aggregates
        )

        (english_select_from_prompt, main_table, selected_fields), agg_phrases = get_english_select_from(
            main_table, selected_fields, use_synonyms_table, use_synonyms_field
        )
        
        (join_table,
         join_fields,
         join_create_statement,
         join_clause_sql,
         english_join_phrase,
         join_condition) = get_sql_join(main_table, main_fields)
        
        include_where = (random.random() < where_clause_probability)
        if include_where:
            (where_fields, where_literals, conditions, sql_where_statement) = get_sql_where(main_fields)
            english_where_prompt = get_english_where(conditions)
        else:
            where_fields = []
            where_literals = []
            conditions = []
            sql_where_statement = ""
            english_where_prompt = ""
        
        include_order_by = (random.random() < order_by_clause_probability)
        if include_order_by:
            (order_by_fields, sql_order_by_statement) = get_sql_order_by(main_fields)
            english_order_by_prompt, order_by_phrase = get_english_order_by(order_by_fields)
        else:
            order_by_fields = []
            sql_order_by_statement = ""
            english_order_by_prompt = ""
            order_by_phrase = ""
        
        full_sql_statement = " ".join([
            sql_select_statement,
            join_clause_sql,
            sql_where_statement,
            sql_order_by_statement
        ]).strip()
        
        full_english_prompt = " ".join([
            english_select_from_prompt,
            english_join_phrase,
            english_where_prompt,
            english_order_by_prompt
        ]).strip()
        
        batch_item = BatchItem(
            command_set=5,
            table_name=TableName(
                name=main_table.name,
                synonym=main_table.synonym,
                use_synonym=main_table.use_synonym
            ),
            table_fields=main_fields,
            create_statement=main_create_statement,
            select=selected_fields,
            order_by=order_by_fields,
            join_table=join_table,
            join_fields=join_fields,
            join_condition=join_condition,
            where=conditions,
            where_fields=where_fields,
            where_literals=where_literals,
            english_prompt=full_english_prompt,
            sql_statement=full_sql_statement,
            order_by_phrase=order_by_phrase,
            agg_phrases=agg_phrases
        )

        batch.append(batch_item)
    
    return batch


def evaluate_cs5_prediction_score(item: BatchItem, predicted_sql_statement: str):
    return (0, 0)


def evaluate_cs5_prediction(item: BatchItem, predicted_sql_statement: str) -> float:
    return 0


def evaluate_cs5_predictions(items, predicted_sql_statements) -> float:
    total_accuracy = 0.0
    for i, item in enumerate(items):
        accuracy = evaluate_cs5_prediction(item, predicted_sql_statements[i])
        total_accuracy += accuracy
    return total_accuracy / len(items)
