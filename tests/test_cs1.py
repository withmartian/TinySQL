import unittest

from training_data import (get_sql_table_names, get_sql_field_names, get_sql_create_table,
                            get_sql_select_from, get_english_select_from_phrases, get_english_select_from_phrase,
                            generate_cs1, evaluate_cs1_prediction)


# Command Set 1 = SELECT, FROM
# Refer https://docs.google.com/document/d/1HZMqWJA5qw8TFhyk8j3WB573ec-8bKMdrE4TnV4-qgc/
class TestCommandSet1(unittest.TestCase):


    def test_table_names(self):
        
        table_names = get_sql_table_names()
        
        print( "# Table names:", len(table_names), "sample:", table_names[23] )
 

    def test_field_names(self):
        
        field_names_and_types = get_sql_field_names()
        
        print( "# Field names:", len(field_names_and_types) )


    def test_create_table(self):
        
        table_names = get_sql_table_names()
        field_names_and_types = get_sql_field_names()       

        (_, _, create_statement) = get_sql_create_table(table_names, field_names_and_types, 3, 9)
        print("Sample Create:", create_statement)


    def test_select_from(self):
        
        table_names = get_sql_table_names()
        field_names_and_types = get_sql_field_names()       

        (table_name, all_fields, _) = get_sql_create_table(table_names, field_names_and_types, 2, 12)

        (_, sql_select_statement) = get_sql_select_from(table_name, all_fields)

        print("Sample Select:", sql_select_statement)


    def test_english_select_from_phrases(self):
        _ = get_english_select_from_phrases()


    def test_get_english_select_from(self):

        table_names = get_sql_table_names()
        field_names_and_types = get_sql_field_names()       

        (table_name, all_fields, _) = get_sql_create_table(table_names, field_names_and_types, 2, 12)

        (selected_fields, _) = get_sql_select_from(table_name, all_fields)

        english_select_from = get_english_select_from_phrase(table_name, selected_fields)

        print( "English select:", english_select_from )


    def test_generate_cs1(self):
            
        batch_size = 50
        answer = generate_cs1(batch_size)
        
        for i in range(batch_size):

            accuracy = evaluate_cs1_prediction(answer[i], answer[i].sql_statement)

            if i == 4:
                answer[i].print()
                print( "Accuracy:", accuracy )

            # The "ground truth" should score 100%
            assert(accuracy == 1)
