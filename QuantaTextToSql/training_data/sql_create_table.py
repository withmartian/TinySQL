import random
from typing import List, Tuple

from .fragments.table_names import get_sql_table_name
from .fragments.field_names import get_sql_table_fields


def get_sql_create_table(min_cols, max_cols) -> Tuple[str, List[Tuple[str, str]], str]:
    """
    Generates a random SQL CREATE statement using provided field names.
        
    Returns:
        tuple: (
            str: A random table name,
            list of tuples: [(field_name, chosen_type), ...],
            str: A SQL CREATE statement with randomly chosen columns
        )
    """

    # Select random table name
    table_name = get_sql_table_name()
    
    # Determine number of columns 
    num_columns = random.randint(min_cols, max_cols)
    
    # Select random fields and ensure no duplicates
    selected_fields = get_sql_table_fields(table_name, num_columns)
    
    # Build the CREATE TABLE statement
    sql_parts = [f"CREATE TABLE {table_name} ("]
    
    # Add columns
    column_definitions = []
    for field in selected_fields:
        column_definitions.append(f"    {field.name} {field.type}")
       
    # Combine all parts
    sql_parts.append(",\n".join(column_definitions))
    sql_parts.append(")")
    
    return (table_name, selected_fields, "\n".join(sql_parts))
