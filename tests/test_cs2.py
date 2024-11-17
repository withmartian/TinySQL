import unittest

from QuantaTextToSql.training_data import (generate_cs2, evaluate_cs2_prediction)


# Command Set 2 = ORDER BY ASC, DESC
# Refer https://docs.google.com/document/d/1HZMqWJA5qw8TFhyk8j3WB573ec-8bKMdrE4TnV4-qgc/
class TestCommandSet2(unittest.TestCase):

    # The "ground truth" should score 100%  
    def test_generate_cs2_ground_truth(self):
            
        batch_size = 3000
        answer = generate_cs2(batch_size) 
        for i in range(batch_size):

            prediction = answer[i].sql_statement
            accuracy = evaluate_cs2_prediction(answer[i], prediction)
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
    def test_generate_cs2_unrecognized_words(self):
        threshold = 0.9
        max_accuracy = 0.0

        batch_size = 1000
        answer = generate_cs2(batch_size)  
        for i in range(batch_size):

            unrecognized_words = " I must not fear. Fear is the mind-killer. "

            prediction = answer[i].sql_statement + unrecognized_words
            accuracy = evaluate_cs2_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

            accuracy = evaluate_cs2_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

        print(f"Max Accuracy: {max_accuracy:.2f}")

    # Duplicated good text should score less than 100% as it is unnecessarily verbose
    def test_generate_cs2_duplicate(self):
        threshold = 0.92
        max_accuracy = 0.0  
            
        batch_size = 1000
        answer = generate_cs2(batch_size)  
        for i in range(batch_size):

            prediction = answer[i].sql_statement + " " + answer[i].sql_statement
            accuracy = evaluate_cs2_prediction(answer[i], prediction) 
            max_accuracy = self.include_prediction(i, prediction, accuracy, threshold, max_accuracy)

        print(f"Max Accuracy: {max_accuracy:.2f}")
     
