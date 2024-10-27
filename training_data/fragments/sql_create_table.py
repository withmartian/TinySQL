import random
from typing import Dict, List, Tuple


def get_sql_create_table(table_names, field_names_and_types: Dict[str, List[str]], min_cols, max_cols) -> Tuple[str, List[Tuple[str, str]], str]:
    """
    Generates a random SQL CREATE statement using provided field names.
    
    Args:
        field_names_and_types (dict): Dictionary of field names and their possible data types
        
    Returns:
        tuple: (
            str: A random table name,
            list of tuples: [(field_name, chosen_type), ...],
            str: A SQL CREATE statement with 2-12 randomly chosen columns
        )
    """

    # Select random table name
    table_name = random.choice(table_names)
    
    # Determine number of columns 
    num_columns = random.randint(min_cols, max_cols)
    
    # Select random fields and ensure no duplicates
    selected_fields = random.sample(list(field_names_and_types.keys()), num_columns)
    
    # Build the CREATE TABLE statement
    sql_parts = [f"CREATE TABLE {table_name} ("]
    
    # Add columns
    column_definitions = []
    for field in selected_fields:
        # Select a random data type from the available options for this field
        data_type = random.choice(field_names_and_types[field])
        column_definitions.append(f"    {field} {data_type}")
    
    # Add primary key if 'id' is in selected fields
    if 'id' in selected_fields:
        idx = selected_fields.index('id')
        column_definitions[idx] += " PRIMARY KEY"
    
    # Combine all parts
    sql_parts.append(",\n".join(column_definitions))
    sql_parts.append(")")
    
    return (table_name, selected_fields, "\n".join(sql_parts))
