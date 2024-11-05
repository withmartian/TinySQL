import unittest

from training_data import get_sql_create_table, get_sql_select_from, get_english_select_from, generate_cs1, evaluate_cs1_prediction


# Command Set 1 = SELECT, FROM
# Refer https://docs.google.com/document/d/1HZMqWJA5qw8TFhyk8j3WB573ec-8bKMdrE4TnV4-qgc/
class TestCommandSet1(unittest.TestCase):


    def test_create_table(self):

        (_, _, create_statement) = get_sql_create_table(3, 9)

        print("Sample Create:", create_statement)


    def test_select_from(self):
     
        (table_name, table_fields, _) = get_sql_create_table(2, 12)

        (_, sql_select_statement) = get_sql_select_from(table_name, table_fields, False)

        print("Sample Select:", sql_select_statement)
     
    
    def test_get_english_select_from(self): 

        (table_name, table_fields, _) = get_sql_create_table(2, 12)

        (select_fields, _) = get_sql_select_from(table_name, table_fields, False)

        english_select_from = get_english_select_from(table_name, select_fields)

        print( "English select:", english_select_from )


    def test_generate_cs1(self):
            
        batch_size = 3000
        answer = generate_cs1(batch_size)
        
        for i in range(batch_size):

            accuracy = evaluate_cs1_prediction(answer[i], answer[i].sql_statement)

            if(i == 4) or (accuracy < 1):
                print("Command set:", answer[i].command_set)                
                print("Context:", answer[i].create_statement)
                print("Prompt:", answer[i].english_prompt)
                print("SQL:", answer[i].sql_statement)
                print("Accuracy:", accuracy)

            if accuracy < 1:
                accuracy = evaluate_cs1_prediction(answer[i], answer[i].sql_statement)

            # The "ground truth" should score 100%              
            assert(accuracy == 1)
