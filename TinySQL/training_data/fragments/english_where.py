def get_english_where(conditions):
    english_conditions = []
    operator_words = {
        '=': "equal to",
        '>': "greater than",
        '<': "less than",
        '>=': "greater than or equal to",
        '<=': "less than or equal to",
        'LIKE': "containing"
    }
    for cond in conditions:
        parts = cond.split()
        if len(parts) >= 3:
            field = parts[0]
            operator = parts[1]
            value = " ".join(parts[2:])
            op_word = operator_words.get(operator, operator)
            english_conditions.append(f"{field} is {op_word} {value}")
    english_where = "where " + " and ".join(english_conditions)
    return english_where
