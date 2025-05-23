import random
from typing import List, Tuple

from .fragments.table_names import get_sql_table_name
from .fragments.field_names import get_sql_table_fields
from .fragments.models import TableName, TableField, trim_newlines_and_multiple_spaces


def get_sql_create_table_from_selected_fields(table_name : TableName, selected_fields : List[TableField]) -> Tuple[TableName, List[Tuple[str, str]], str]:
    """
    Generates a random SQL CREATE statement using provided field names.
        
    Returns:
        tuple: (
            TableName: A random table name with synonym
            list of tuples: [(field_name, chosen_type), ...],
            str: A SQL CREATE statement with randomly chosen columns
        )
    """
    
    # Build the CREATE TABLE statement
    sql_parts = [f"CREATE TABLE {table_name.name} ( "]
    
    # Add columns
    column_definitions = []
    for field in selected_fields:
        column_definitions.append(f" {field.name} {field.type}")
       
    # Combine all parts
    sql_parts.append(", ".join(column_definitions))
    sql_parts.append(" )")
    sql_statement = trim_newlines_and_multiple_spaces(" ".join(sql_parts))

    return (table_name, selected_fields, sql_statement)



def get_sql_create_table(min_cols, max_cols) -> Tuple[TableName, List[Tuple[str, str]], str]:
    """
    Generates a random SQL CREATE statement with required number of columns.
        
    Returns:
        tuple: (
            TableName: A random table name and synonym,
            list of tuples: [(field_name, chosen_type), ...],
            str: A SQL CREATE statement with randomly chosen columns
        )
    """

    # Select random table name and synonym
    table_name = get_sql_table_name()
    
    # Determine number of columns 
    num_columns = random.randint(min_cols, max_cols)
    
    # Select random fields and ensure no duplicates
    selected_fields = get_sql_table_fields(table_name, num_columns)
    
    return get_sql_create_table_from_selected_fields(table_name, selected_fields)

