import unittest

from TinySQL.training_data.generate_cs1 import generate_cs1, evaluate_cs1_prediction
from TinySQL.training_data.generate_cs2 import generate_cs2, evaluate_cs2_prediction
from TinySQL.training_data.generate_cs3 import generate_cs3, evaluate_cs3_prediction
from TinySQL.training_data import generate_dataset

class TestGenerate(unittest.TestCase):

    def test_generate(self):
        batch_size = 16
        push_to_hf = False

        use_synonyms = False
        generate_dataset(batch_size, generate_cs1, evaluate_cs1_prediction, "withmartian/cs1_dataset", use_synonyms, push_to_hf)
        generate_dataset(batch_size, generate_cs2, evaluate_cs2_prediction, "withmartian/cs2_dataset", use_synonyms, push_to_hf)
        generate_dataset(batch_size, generate_cs3, evaluate_cs3_prediction, "withmartian/cs3_dataset", use_synonyms, push_to_hf)

        use_synonyms = True
        generate_dataset(batch_size, generate_cs1, evaluate_cs1_prediction, "withmartian/cs1_dataset_synonyms", use_synonyms, push_to_hf)
        generate_dataset(batch_size, generate_cs2, evaluate_cs2_prediction, "withmartian/cs2_dataset_synonyms", use_synonyms, push_to_hf)
        generate_dataset(batch_size, generate_cs3, evaluate_cs3_prediction, "withmartian/cs3_dataset_synonyms", use_synonyms, push_to_hf)
