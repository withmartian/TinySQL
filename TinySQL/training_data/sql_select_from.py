import random
from .fragments.field_names import get_sql_select_fields
from .fragments.models import TableName, SelectField, TableField, trim_newlines_and_multiple_spaces


def get_sql_select_from_selected_fields(table_name : TableName, selected_fields):
    """
    Generates a random SQL SELECT statement using provided field names.
    
    Args:
        table_name (TableName): Name of the table to select from and a synonym
        table_fields (list): List of field names and their data type
        use_aggregates (bool): Whether to include aggregate functions
        
    Returns:
        tuple: (selected_fields, sql_statement)
    """
     
    formatted_fields = []  
    for field in selected_fields:
        if isinstance(field, SelectField):
            formatted_fields.append(field.sql)
        elif isinstance(field, TableField):
            formatted_fields.append(field.name)

    if len(formatted_fields) == 1:
        sql = f"SELECT {formatted_fields[0]} FROM {table_name.name}"
    else:
        sql = "SELECT " + ", ".join(formatted_fields) + f" FROM {table_name.name}"
    
    sql = trim_newlines_and_multiple_spaces(sql)

    return (selected_fields, sql)


def get_sql_select_from(table_name : TableName, table_fields, use_aggregates: bool):
    """
    Generates a random SQL SELECT statement using provided field names.
    
    Args:
        table_name (TableName): Name of the table to select from and its synonym
        table_fields (list): List of field names and their data type
        use_aggregates (bool): Whether to include aggregate functions
        
    Returns:
        tuple: (selected_fields, sql_statement)
    """

    # Get number of columns to select 
    num_columns = random.randint(1, len(table_fields))
    
    selected_fields = get_sql_select_fields(table_fields, num_columns, use_aggregates)
     
    return get_sql_select_from_selected_fields(table_name, selected_fields)
