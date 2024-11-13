import unittest

from QuantaTextToSql.training_data import (generate_cs3, evaluate_cs3_prediction)


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
        for i in range(batch_size):

            # Using answer[i], create a JSON style output
            json_answer = {
                "table_name": answer[i].table_name,
                "table_fields": [table_field.name for table_field in answer[i].table_fields],
                "select": [select_field.name for select_field in answer[i].select],
                "order_by": [order_field.name for order_field in answer[i].order_by],
                "english_prompt": answer[i].english_prompt
            }
            prediction = str(json_answer)
            accuracy = evaluate_cs3_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

            prediction = "{ """ + answer[i].table_name+ " : [ "
            for table_field in answer[i].table_fields:
                prediction += " """ + table_field.name + " : ""somedata"", "
            prediction += " ] }"
            accuracy = evaluate_cs3_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)       

        print(f"Max Accuracy: {max_accuracy:.2f}")