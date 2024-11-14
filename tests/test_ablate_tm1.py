import unittest

from QuantaTextToSql.ablate import load_tm1, collect_tm1_activations, ablate_tm1

#import QuantaMechInterp as qmi
#print(qmi.__file__) 


class TestAblateTM1(unittest.TestCase):

    def setUp(self):
        self.tokenizer, self.model = load_tm1()

        collect_tm1_activations(self.model, self.tokenizer)

    def test_ablate_tm1_layer(self):
        ablate_tm1(self.tokenizer, self.model, node_type="layer", layer_index=0)
        ablate_tm1(self.tokenizer, self.model, node_type="layer", layer_index=1)

    def test_ablate_tm1_mlp(self):
        ablate_tm1(self.tokenizer, self.model, node_type="mlp", layer_index=0)
        ablate_tm1(self.tokenizer, self.model, node_type="mlp", layer_index=1)

    def test_ablate_tm1_head(self):
        ablate_tm1(self.tokenizer, self.model, node_type="head", layer_index=0, head_index=5)
        ablate_tm1(self.tokenizer, self.model, node_type="head", layer_index=1, head_index=4)

    def tearDown(self):
        del self.tokenizer
        del self.model