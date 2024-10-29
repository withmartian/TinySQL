import random


def get_english_order_by_phrases():
    """
    Returns a list of tuples containing:
    1. Common English phrases indicating sorting/ordering intent
    2. The corresponding SQL sort direction (ASC or DESC)
    
    These map natural language to SQL's ORDER BY clause with direction.
    """
    return [
        # Ensure model sees correct answer tokens during training
        ("ORDER BY", "ASC"),
        ("ORDER BY ASC", "ASC"),
        ("ORDER BY DESC", "DESC"),

        # Simple ordering phrases (default to ASC as it's the normal expectation)
        ("sorted by", "ASC"),
        ("sort by", "ASC"),
        ("ordered by", "ASC"),
        ("order by", "ASC"),
        ("arrange by", "ASC"),
        ("arranged by", "ASC"),
        ("ranked by", "ASC"),
        ("rank by", "ASC"),
        
        # Explicit directional phrases
        ("in ascending order of", "ASC"),
        ("in descending order of", "DESC"),
        ("in order of", "ASC"),
        ("from lowest to highest", "ASC"),
        ("from highest to lowest", "DESC"),
        ("from smallest to largest", "ASC"),
        ("from largest to smallest", "DESC"),
        ("from oldest to newest", "ASC"),
        ("from newest to oldest", "DESC"),
        ("from least to most", "ASC"),
        ("from most to least", "DESC"),
        
        # Comparative phrases (default to ASC)
        ("organized by", "ASC"),
        ("listed by", "ASC"),
        ("grouped by", "ASC"),
        ("categorized by", "ASC"),
        ("classified by", "ASC"),
        ("structured by", "ASC"),
        ("sequenced by", "ASC"),
        
        # Time-based phrases
        ("chronologically by", "ASC"),
        ("in chronological order of", "ASC"),
        ("in reverse chronological order of", "DESC"),
        ("date ordered by", "ASC"),
        ("time ordered by", "ASC"),
        ("ordered by date of", "ASC"),
        ("sorted by time of", "ASC"),
        ("most recent", "DESC"),
        ("latest", "DESC"),
        ("newest", "DESC"),
        ("oldest", "ASC"),
        
        # Priority phrases (typically DESC as higher priority usually comes first)
        ("prioritized by", "DESC"),
        ("priority ordered by", "DESC"),
        ("ranked in terms of", "DESC"),
        ("ordered according to", "ASC"),
        ("sorted according to", "ASC"),
        ("arranged according to", "ASC"),
        
        # Numeric phrases
        ("numerically by", "ASC"),
        ("in numerical order of", "ASC"),
        ("in reverse numerical order of", "DESC"),
        ("sorted numerically by", "ASC"),
        ("ordered numerically by", "ASC"),
        
        # Alphabetical phrases
        ("alphabetically by", "ASC"),
        ("in alphabetical order of", "ASC"),
        ("in reverse alphabetical order of", "DESC"),
        ("sorted alphabetically by", "ASC"),
        ("ordered alphabetically by", "ASC"),
        ("a to z by", "ASC"),
        ("z to a by", "DESC"),
        
        # Natural language phrases (typically DESC as people often want highest/most first)
        ("with the highest", "DESC"),
        ("with the lowest", "ASC"),
        ("starting with the highest", "DESC"),
        ("starting with the lowest", "ASC"),
        ("beginning with the most", "DESC"),
        ("beginning with the least", "ASC"),
        ("showing first the highest", "DESC"),
        ("showing first the lowest", "ASC"),
        ("top", "DESC"),
        ("bottom", "ASC"),
        ("best", "DESC"),
        ("worst", "ASC"),
        ("most", "DESC"),
        ("least", "ASC")
    ]


def get_english_order_by_phrase(field_names: list[str], asc : bool) -> str:
    """
    Generates both an English phrase and corresponding SQL statement for a SELECT query.
    
    Args:
        field_names (list[str]): List of field names to select
        asc bool: Ascending or descending order
        
    Returns:
        str: english_phrase
    """
    # Get list of template phrases
    templates = get_english_order_by_phrases()
    
    # Filter templates to match the asc/desc preference
    filtered_templates = [phrase for phrase, direction in templates if (direction == "ASC") == asc]
    
    # Select random template
    template = random.choice(filtered_templates)
    
    # Format fields for English 
    english_fields = ", ".join(field_names[:-1]) + " and " + field_names[-1] if len(field_names) > 1 else field_names[0]
    
    # Create English phrase
    english = template + " " + english_fields
    
    return english