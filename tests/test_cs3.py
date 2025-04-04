import unittest

from TinySQL.training_data import (generate_cs3, evaluate_cs3_prediction, generate_csn)


# Command Set 3 = MAX, MIN, AVG, SUM, COUNT
# Refer https://docs.google.com/document/d/1HZMqWJA5qw8TFhyk8j3WB573ec-8bKMdrE4TnV4-qgc/
class TestCommandSet3(unittest.TestCase):

    # The "ground truth" should score 100%  
    def test_generate_cs3_ground_truth(self):
            
        batch_size = 3000
        answer = generate_cs3(batch_size) 
        for i in range(batch_size):

            prediction = answer[i].sql_statement
            accuracy = evaluate_cs3_prediction(answer[i], prediction)
            if accuracy < 1:
                print("Example:", i)                
                print("Prediction", prediction)
                print("Accuracy:", accuracy)  
            assert(accuracy == 1)

    def include_prediction(self, i, prediction, accuracy, threshold, max_accuracy):
        if accuracy >= threshold:        
            print("Example:", i)                
            print("Prediction", prediction)
            print("Accuracy:", accuracy)
            assert(False)

        return max( accuracy, max_accuracy )
    
    # Unrecognized words should score less than 100%
    def test_generate_cs3_unrecognized_words(self):
        threshold = 0.91
        max_accuracy = 0.0

        batch_size = 1000
        answer = generate_cs3(batch_size)  
        for i in range(batch_size):

            unrecognized_words = " I must not fear. Fear is the mind-killer. "
            prediction = answer[i].sql_statement + unrecognized_words
            accuracy = evaluate_cs3_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

            prediction = unrecognized_words + answer[i].sql_statement        
            accuracy = evaluate_cs3_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

        print(f"Max Accuracy: {max_accuracy:.2f}")

    # Duplicated good text should score less than 100% as it is unnecessarily verbose
    def test_generate_cs3_duplicate(self):
        threshold = 0.982
        max_accuracy = 0.0

        batch_size = 1000
        answer = generate_cs3(batch_size)  
        for i in range(batch_size):

            prediction = answer[i].sql_statement + " " + answer[i].sql_statement
            accuracy = evaluate_cs3_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

        print(f"Max Accuracy: {max_accuracy:.2f}")

    # JSON style output should score less than 100%
    def test_generate_cs3_json_output(self):
        threshold = 0.75
        max_accuracy = 0.0

        batch_size = 1000
        answer = generate_cs3(batch_size)  
        print(len(answer))
        for i in range(batch_size):

            # Using answer[i], create a JSON style output
            json_answer = {
                "table_name": answer[i].table_name.name,
                "table_name_synonym": answer[i].table_name.synonym, # May be identical to table_name 
                "table_name_use_synonym": answer[i].table_name.use_synonym,
                "table_fields": [table_field.name for table_field in answer[i].table_fields],
                "table_field_synonyms": [table_field.synonym for table_field in answer[i].table_fields], # May be identical to table_fields                 
                "table_field_use_synonyms": [table_field.use_synonym for table_field in answer[i].table_fields],               
                "select": [select_field.name for select_field in answer[i].select],
                "order_by": [order_field.name for order_field in answer[i].order_by],
                "english_prompt": answer[i].english_prompt
            }
            prediction = str(json_answer)
            accuracy = evaluate_cs3_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

            prediction = "{ """ + answer[i].table_name.name + " : [ "
            for table_field in answer[i].table_fields:
                prediction += " """ + table_field.name + " : ""somedata"", "
            prediction += " ] }"
            accuracy = evaluate_cs3_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)       

        print(f"Max Accuracy: {max_accuracy:.2f}")


    def test_generate_csn(self):
        generate_csn(5, 1, 0.9, False, 2, 6, False, False)
        generate_csn(5, 2, 0.9, False, 3, 3, False, False)
        generate_csn(5, 3, 0.9, True, 1, 4, False, False)

        generate_csn(5, 1, 0.9, False, 2, 6, True, True)
        generate_csn(5, 2, 0.9, False, 3, 3, True, True)
        generate_csn(5, 3, 0.9, True, 1, 4, True, True)       


    def test_cs3_statistics(self):
        batch_size = 1000
        answer = generate_cs3(batch_size, use_synonyms_field=True)

        max_english_chars = max_english_words = 0
        max_create_chars = max_create_words = 0
        max_sql_chars = max_sql_words = 0

        for item in answer:
            english = item.english_prompt
            create = item.create_statement
            sql = item.sql_statement

            max_english_chars = max(max_english_chars, len(english))
            max_english_words = max(max_english_words, len(english.split()))

            max_create_chars = max(max_create_chars, len(create))
            max_create_words = max(max_create_words, len(create.split()))

            max_sql_chars = max(max_sql_chars, len(sql))
            max_sql_words = max(max_sql_words, len(sql.split()))

        print(f"Sample: {answer[0]}")
        print(f"Max english_prompt: {max_english_words} words, {max_english_chars} characters")
        print(f"Max create_statement: {max_create_words} words, {max_create_chars} characters")
        print(f"Max sql_statement: {max_sql_words} words, {max_sql_chars} characters")        
