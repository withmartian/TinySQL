import random
from .sql_create_table import get_sql_create_table


def get_sql_join(main_table, main_fields, min_cols=1, max_cols=4):
    (join_table, join_fields, join_create_statement) = get_sql_create_table(
        min_cols=min_cols, max_cols=max_cols
    )

    def base_type(sql_type: str) -> str:
        return sql_type.upper().split("(")[0]

    matching_pairs = []
    for field_main in main_fields:
        main_base = base_type(field_main.type)
        for field_join in join_fields:
            join_base = base_type(field_join.type)
            if main_base == join_base:
                matching_pairs.append((field_main, field_join))

    if matching_pairs:
        (field_main, field_join) = random.choice(matching_pairs)
        join_clause_sql = (
            f"JOIN {join_table.name} "
            f"ON {main_table.name}.{field_main.name} = {join_table.name}.{field_join.name}"
        )
        english_join_phrase = (
            f"join with {join_table.synonym if join_table.use_synonym else join_table.name} "
            f"on {field_main.name} equals {field_join.name}"
        )
        join_condition = (field_main, field_join)
    else:
        join_table = None
        join_fields = []
        join_create_statement = ""
        join_clause_sql = ""
        english_join_phrase = ""
        join_condition = None

    return (
        join_table,
        join_fields,
        join_create_statement,
        join_clause_sql,
        english_join_phrase,
        join_condition,
    )
