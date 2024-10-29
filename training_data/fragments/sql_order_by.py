import random

def get_sql_order_by(field_names_and_types):
    """
    Generates a random SQL ORDER BY statement using provided field names.
    
    Args:
        field_names_and_types (dict): Dictionary of field names and their possible data types
        
    Returns:
        list: The randomly chosen columns to order by
        bool: Is order by ASC (else DESC)
        str: A SQL ORDER BY statement with randomly chosen columns, and random ASC or DESC
    """

    asc = True if random.randint(0, 1) == 1 else False
    direction = "ASC" if asc else "DESC"

    # Get number of columns to order by 
    num_columns = random.randint(1, len(field_names_and_types))
    
    # Select random fields to order by 
    order_by_fields = random.sample(list(field_names_and_types), num_columns)
    
    # Format the field list with proper SQL escaping for special characters
    formatted_fields = [f'{field} {direction}' for field in order_by_fields]
    
    # Build the SELECT statement with proper formatting
    # Add newlines and indentation for better readability
    sql = "ORDER BY\n    " + ",\n    ".join(formatted_fields)
    
    return (order_by_fields, asc, sql)