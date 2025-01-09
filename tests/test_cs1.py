import unittest

from TinySQL.training_data import get_sql_create_table, get_sql_select_from, get_english_select_from, generate_cs1, evaluate_cs1_prediction


# Command Set 1 = SELECT, FROM
# Refer https://docs.google.com/document/d/1HZMqWJA5qw8TFhyk8j3WB573ec-8bKMdrE4TnV4-qgc/
class TestCommandSet1(unittest.TestCase):

    def test_create_table(self):

        (_, _, create_statement) = get_sql_create_table(3, 9)

        #print("Sample Create:", create_statement)

    def test_select_from(self):
     
        (table_name, table_fields, _) = get_sql_create_table(2, 12)

        (_, sql_select_statement) = get_sql_select_from(table_name, table_fields, False)

        #print("Sample Select:", sql_select_statement)
    
    def test_get_english_select_from(self): 

        (table_name, table_fields, _) = get_sql_create_table(2, 12)

        (select_fields, _) = get_sql_select_from(table_name, table_fields, False)

        english_select_from = get_english_select_from(table_name, select_fields, False)
        print( "English select (no synonyms):", english_select_from )

        english_select_from = get_english_select_from(table_name, select_fields, True)
        print( "English select (with synonyms):", english_select_from )

    # The "ground truth" should score 100%  
    def test_generate_cs1_ground_truth(self):
            
        i = 0
        batch_size = 3000
        batch = generate_cs1(batch_size) 
        for batch_item in batch:
            prediction = batch_item.sql_statement
            accuracy = evaluate_cs1_prediction(batch_item, prediction)
            if accuracy < 1:
                print("Example :", i)                
                print("Table   :", batch_item.table_name.name, "with synonym:", batch_item.table_name.synonym)                
                print("Answer  :", batch_item.sql_statement)                
                print("Predicts:", prediction)
                print("Accuracy:", accuracy)      
            assert(accuracy == 1)
            i += 1

    def include_prediction(self, i, prediction, accuracy, threshold, max_accuracy):
        if accuracy >= threshold:        
            print("Example:", i)                
            print("Prediction", prediction)
            print("Accuracy:", accuracy)
            assert(False)

        return max( accuracy, max_accuracy )
    
    # Unrecognized words should score less than 100%
    def test_generate_cs1_unrecognized_words(self):
        threshold = 0.9
        max_accuracy = 0.0
            
        batch_size = 1000
        answer = generate_cs1(batch_size)  
        for i in range(batch_size):

            unrecognized_words = " I must not fear. Fear is the mind-killer. "

            prediction = answer[i].sql_statement + unrecognized_words
            accuracy = evaluate_cs1_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

            prediction = unrecognized_words + answer[i].sql_statement
            accuracy = evaluate_cs1_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

        print(f"Max Accuracy: {max_accuracy:.2f}")

    # Duplicated good text should score less than 100% as it is unnecessarily verbose
    def test_generate_cs1_duplicate(self):
        threshold = 0.9
        max_accuracy = 0.0

        batch_size = 1000
        answer = generate_cs1(batch_size)  
        for i in range(batch_size):

            prediction = answer[i].sql_statement + " " + answer[i].sql_statement
            accuracy = evaluate_cs1_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

        print(f"Max Accuracy: {max_accuracy:.2f}")