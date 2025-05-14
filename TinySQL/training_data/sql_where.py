import random
from .fragments.where_utilities import (
        random_numeric_value, random_text_value,
        random_date_value, random_boolean_value,
        random_uuid_value, random_blob_value,
        random_json_value
        )


def get_sql_where(table_fields, max_conditions=3):
    num_possible = min(max_conditions, len(table_fields))
    num_conditions = random.randint(1, num_possible)
    selected_fields = random.sample(table_fields, num_conditions)

    condition_operators = {
        "INTEGER": ["=", ">", "<", ">=", "<="],
        "BIGINT": ["=", ">", "<", ">=", "<="],
        "DECIMAL": ["=", ">", "<", ">=", "<="],
        "NUMERIC": ["=", ">", "<", ">=", "<="],
        "FLOAT": ["=", ">", "<", ">=", "<="],
        "DOUBLE": ["=", ">", "<", ">=", "<="],
        "VARCHAR": ["LIKE"],
        "CHAR": ["LIKE"],
        "TEXT": ["LIKE"],
        "DATE": ["=", ">", "<", ">=", "<="],
        "DATETIME": ["=", ">", "<", ">=", "<="],
        "TIMESTAMP": ["=", ">", "<", ">=", "<="],
        "BOOLEAN": ["="],
        "UUID": ["="],
        "BLOB": ["="],
        "JSON": ["="],
        "JSONB": ["="]
    }

    conditions = []
    sql_conditions = []
    for field in selected_fields:
        field_type = field.type.upper()
        if "(" in field_type:
            base_type = field_type.split("(")[0]
        else:
            base_type = field_type

        operators = condition_operators.get(base_type, ["="])
        operator = random.choice(operators)

        if base_type in ["INTEGER", "BIGINT", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE"]:
            value = random_numeric_value()
        elif base_type in ["VARCHAR", "CHAR", "TEXT"]:
            value = random_text_value()
        elif base_type in ["DATE", "DATETIME", "TIMESTAMP"]:
            value = random_date_value()
        elif base_type == "BOOLEAN":
            value = random_boolean_value()
        elif base_type == "UUID":
            value = random_uuid_value()
        elif base_type == "BLOB":
            value = random_blob_value()
        elif base_type in ["JSON", "JSONB"]:
            value = random_json_value()
        else:
            value = random_numeric_value()

        condition = f"{field.name} {operator} {value}"
        conditions.append(condition)
        sql_conditions.append(condition)
    sql_where_statement = "WHERE " + " AND ".join(sql_conditions)
    return conditions, sql_where_statement
