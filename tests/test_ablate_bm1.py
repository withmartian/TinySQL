import unittest

from QuantaTextToSql.ablate import load_bm1, collect_bm1_activations, ablate_bm1

#import QuantaMechInterp as qmi
#print(qmi.__file__) 


class TestAblateBM1(unittest.TestCase):

    def setUp(self):
        self.tokenizer, self.model = load_bm1()

        collect_bm1_activations(self.model, self.tokenizer)

    def test_ablate_bm1_layer(self):
        ablate_bm1(self.tokenizer, self.model, node_type="layer", layer_index=0)
        ablate_bm1(self.tokenizer, self.model, node_type="layer", layer_index=1)

    def test_ablate_bm1_mlp(self):
        ablate_bm1(self.tokenizer, self.model, node_type="mlp", layer_index=0)
        ablate_bm1(self.tokenizer, self.model, node_type="mlp", layer_index=1)

    def test_ablate_bm1_head(self):
        ablate_bm1(self.tokenizer, self.model, node_type="attention_head", layer_index=0, head_index=5)
        ablate_bm1(self.tokenizer, self.model, node_type="attention_head", layer_index=1, head_index=4)

    def tearDown(self):
        del self.tokenizer
        del self.model