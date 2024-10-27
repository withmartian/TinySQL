from training_data import get_sql_table_names, get_sql_field_names, get_sql_create_table, get_sql_select_from, get_english_select_from_phrase


def generate_CS1(batch_size):
    table_names = get_sql_table_names()
    field_names_and_types = get_sql_field_names()       

    answer = []
    for i in range(batch_size):
        (table_name, all_fields, create_table) = get_sql_create_table(table_names, field_names_and_types, 2, 12)

        (selected_fields, sql_select_statement) = get_sql_select_from(table_name, all_fields)

        english_select_from = get_english_select_from_phrase(table_name, selected_fields)

        answer.append([table_name, all_fields, create_table, english_select_from, sql_select_statement])

    return answer
