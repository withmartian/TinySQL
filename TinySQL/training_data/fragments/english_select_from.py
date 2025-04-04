import random
from functools import lru_cache


@lru_cache(maxsize=1)
def get_english_select_from_phrases():
    """
    Returns a list of templated English phrases commonly used to request SQL SELECT operations.
    Uses [table] and [fields] as replacement tokens.

    """
    return [
        # Ensure model sees correct answer tokens during training
        "SELECT [fields] FROM [table]",

        # Direct commands
        "Show me [fields] from [table]",
        "Show [fields] from [table]",
        "Display [fields] from [table]",
        "List [fields] from [table]",
        "Get [fields] from [table]",
        "Pull up [fields] from [table]",
        "Fetch [fields] from [table]",
        "Extract [fields] from [table]",
        "Return [fields] from [table]",
        "Print out [fields] from [table]",
        
        # Question forms
        "What's the [fields] from [table]?",
        "What are the [fields] in [table]?",
        "What do we have for [fields] in [table]?",
        
        # Polite requests
        "Can you get me [fields] from [table]?",
        "Could you show me [fields] from [table]?",
        "Would you mind getting [fields] from [table]?",
        "Please get me [fields] from [table]",
        "I'd like to see [fields] from [table]",
        
        # Need/Want statements
        "I need to see [fields] from [table]",
        "I want to see [fields] from [table]",
        "I need a list of [fields] from [table]",
        "I need access to [fields] from [table]",
        
        # Action-oriented
        "Look up [fields] from [table]",
        "Find [fields] from [table]",
        "Retrieve [fields] from [table]",
        "Pull out [fields] from [table]",
        "Bring up [fields] from [table]",
        "Check [fields] in [table]",
        "Search for [fields] in [table]",
        "Give me [fields] from [table]",
        "Get me [fields] from [table]",
        
        # Let/Share patterns
        "Let me see [fields] from [table]",
        "Let's see [fields] from [table]",
        "Share [fields] from [table]",
        
        # Table-first patterns
        "From [table] show me [fields]",
        "From [table] display [fields]",
        "From [table] get [fields]",
        "In the [table] table, display [fields]",
        "Looking at [table], I need [fields]",
        "From [table], get me [fields]",
        "Within [table], show [fields]",
        "For the [table], display [fields]",
        "In [table], list [fields]",
        "Looking in [table], show me [fields]",
        "Inside [table], find [fields]",
        "Starting with [table], give me [fields]",
        "Out of [table], pull [fields]",
        "Using [table], display [fields]",
        
        # Simple/direct patterns
        "Just the [fields] from [table] please",
        "[fields] from [table]",
        
        # Run/Output patterns
        "Run a query for [fields] in [table]",
        "Output [fields] from [table]",
        "Get a readout of [fields] from [table]",
        
        # Tell/Read patterns
        "Tell me [fields] from [table]",
        "Read out [fields] from [table]"
    ]


def get_english_select_from_phrase() -> str:

    phrases = get_english_select_from_phrases()    

    phrase = random.choice(phrases)

    return phrase


def get_english_select_from_count():
    return len(get_english_select_from_phrases())