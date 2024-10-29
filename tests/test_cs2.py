import unittest

from training_data import (generate_cs2, evaluate_cs2_prediction, get_english_order_by_phrases)


# Command Set 2 = ORDER BY ASC, DESC
# Refer https://docs.google.com/document/d/1HZMqWJA5qw8TFhyk8j3WB573ec-8bKMdrE4TnV4-qgc/
class TestCommandSet2(unittest.TestCase):

    def test_get_english_order_by_phrases(self):

        english_order_by = get_english_order_by_phrases()
        
        print( "# English Order By phrases:", len(english_order_by) )


    def test_generate_cs2(self):
            
        batch_size = 50
        answer = generate_cs2(batch_size)
        
        for i in range(batch_size):

            accuracy = evaluate_cs2_prediction(answer[i], answer[i].sql_statement)


            answer[i].print()
            print( "Accuracy:", accuracy )

            # The "ground truth" should score 100%
            assert(accuracy == 1)