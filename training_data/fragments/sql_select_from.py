import random

def get_sql_select_from(table_name, field_names_and_types):
    """
    Generates a random SQL SELECT statement using provided field names.
    
    Args:
        field_names_and_types (dict): Dictionary of field names and their possible data types
        
    Returns:
        str: A SQL SELECT statement with 1-8 randomly chosen columns
    """
    # Get number of columns to select 
    num_columns = random.randint(1, len(field_names_and_types))
    
    # Select random fields  
    selected_fields = random.sample(list(field_names_and_types), num_columns)
    
    # Format the field list with proper SQL escaping for special characters
    formatted_fields = [f'{field}' for field in selected_fields]
    
    # Build the SELECT statement with proper formatting
    # Add newlines and indentation for better readability
    sql = "SELECT\n    " + ",\n    ".join(formatted_fields) + "\nFROM " + table_name
    
    return (selected_fields, sql)