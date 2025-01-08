import random
from functools import lru_cache

@lru_cache(maxsize=1)
def get_english_max_phrases():
    """
    Returns a list of tuples containing common English phrases indicating max(field) intent
    These map natural language to SQL's MAX command
    """

    # TODO: Associate each phrase with valid data types. Use that association to improve SQL command validity.  

    return [
        # Ensure model sees correct answer tokens during training
        "max",

        # Direct maximum indicators
        "highest", "maximum", "greatest", "largest", "peak",
        "biggest", "latest", "furthest",

        # Compound phrases
        "most expensive", "most recent",
        "record high", "all-time high",
        "last occurring"
    ]

@lru_cache(maxsize=1)
def get_english_min_phrases():
    """
    Returns a list of common English phrases indicating min(field) intent
    These map natural language to SQL's MIN command
    """
    return [
        # Ensure model sees correct answer tokens during training
        "min",
        
        # Direct minimum indicators
        "lowest", "minimum", "smallest", "least",
        "tiniest", "shortest", "minimal",
        
        # Compound phrases
        "least expensive", "least recent",
        "all-time low", "record low",
        
        # Time-based minimums
        "earliest", "first", "oldest",
        "initial", "starting",
        
        # Bottom indicators
        "bottom", "lowest occurring", "minimal amount"
    ]

@lru_cache(maxsize=1)
def get_english_avg_phrases():
    """
    Returns a list of common English phrases indicating avg(field) intent
    These map natural language to SQL's AVG command
    """
    return [
        # Ensure model sees correct answer tokens during training
        "avg", "average",
        
        # Direct average indicators
        "mean", "typical", "median",
        "middle", "midpoint",
        
        # Compound phrases
        "on average", "typically",
        "usual amount", "normal amount",
        "expected value", "average value",
        
        # Statistical terms
        "arithmetic mean", "expected",
        "nominal", "standard",
        
        # Common phrases
        "usual", "typical amount",
        "generally", "normally"
    ]

@lru_cache(maxsize=1)
def get_english_sum_phrases():
    """
    Returns a list of common English phrases indicating sum(field) intent
    These map natural language to SQL's SUM command
    """
    return [
        # Ensure model sees correct answer tokens during training
        "sum",
        
        # Direct sum indicators
        "total", "sum of", "summation",
        "combined", "aggregate",
        
        # Amount indicators
        "overall amount", "full amount",
        "entire amount", "complete amount",
        
        # Cumulative indicators
        "cumulative", "accumulated",
        "running total", "grand total",
        
        # Combined phrases
        "all together", "in total",
        "added up", "summed up",
        "combined total", "total sum"
    ]

@lru_cache(maxsize=1)
def get_english_count_phrases():
    """
    Returns a list of common English phrases indicating count(field) intent
    These map natural language to SQL's COUNT command
    """
    return [
        # Ensure model sees correct answer tokens during training
        "count",
        
        # Direct count indicators
        "number of", "total number", "count of",
        
        # Question forms
        "how many", 
        
        # Counting phrases
        "tally", "frequency", "occurrence",
        
        # Measurement phrases
        "instances of", "occurrences of",
        "times", "frequency of",
        
        # Total indicators
        "total count",
        "overall count", "complete count"
    ]


# Return an english phrase for one of SUM, AVG, MIN, MAX, COUNT or ""
def get_english_aggregate_phrase():

    # Randomly select an aggregate function
    aggregate_functions = ["SUM", "AVG", "MIN", "MAX", "COUNT", ""]
    aggregate_function = random.choice(aggregate_functions)

    # Get the corresponding English phrases for the selected aggregate function
    if aggregate_function == "SUM":
        phrases = get_english_sum_phrases()
    elif aggregate_function == "AVG":
        phrases = get_english_avg_phrases()
    elif aggregate_function == "MIN":
        phrases = get_english_min_phrases()
    elif aggregate_function == "MAX":
        phrases = get_english_max_phrases()
    elif aggregate_function == "COUNT":
        phrases = get_english_count_phrases()
    else:
        phrases = [""]

    return (aggregate_function, random.choice(phrases))


def get_english_aggregate_count():
    return len(get_english_max_phrases())+len(get_english_min_phrases())+len(get_english_avg_phrases())+len(get_english_sum_phrases())+len(get_english_count_phrases())
