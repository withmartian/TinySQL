import unittest

from training_data import (generate_cs3, evaluate_cs3_prediction)


# Command Set 3 = MAX, MIN, AVG, SUM, COUNT
# Refer https://docs.google.com/document/d/1HZMqWJA5qw8TFhyk8j3WB573ec-8bKMdrE4TnV4-qgc/
class TestCommandSet3(unittest.TestCase):

    def test_generate_cs3(self):
            
        batch_size = 3000
        answer = generate_cs3(batch_size)
        
        for i in range(batch_size):

            accuracy = evaluate_cs3_prediction(answer[i], answer[i].sql_statement)

            if(i == 4) or (accuracy < 1):
                print("Example:", i)                
                print("Context:", answer[i].create_statement)
                print("Prompt:", answer[i].english_prompt)
                print("SQL:", answer[i].sql_statement)
                print("Accuracy:", accuracy)

            # The "ground truth" should score 100%              
            assert(accuracy == 1)