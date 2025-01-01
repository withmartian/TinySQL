import random
from typing import Tuple
from .fragments.models import OrderField, TableField, trim_newlines_and_multiple_spaces


def get_sql_order_by(table_fields : list[TableField]) -> Tuple[list[OrderField], str]:
    """
    Generates a random SQL ORDER BY statement using provided field names.
    
    Args:
        field_names_and_types (dict): Dictionary of field names and their possible data types
        
    Returns:
        list[OrderField]: The randomly chosen columns to order by
        str: A SQL ORDER BY statement with randomly chosen columns, and random ASC or DESC
    """

    asc = True if random.randint(0, 1) == 1 else False
    direction = "ASC" if asc else "DESC"

    # Get number of columns to order by 
    num_columns = random.randint(1, len(table_fields))
    
    # Select N fields to order by, in a random order 
    selected_fields = random.sample(table_fields, num_columns)

    order_by_fields = [OrderField(name=field.name, asc=asc) for field in selected_fields]
    
    # Format the field list with proper SQL escaping for special characters
    formatted_fields = [f'{field.name} {direction}' for field in order_by_fields]
    
    # Build the SELECT statement with proper formatting
    # Add newlines and indentation for better readability
    sql = "ORDER BY " + ", ".join(formatted_fields)
    
    sql = trim_newlines_and_multiple_spaces(sql)

    return (order_by_fields, sql)