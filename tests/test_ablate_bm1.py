import unittest

from QuantaTextToSql.ablate import load_bm1, ablate_bm1_layer, ablate_bm1_mlp #, ablate_bm1_head

#import QuantaMechInterp as qmi
#print(qmi.__file__) 


class TestAblateBM1(unittest.TestCase):

    def setUp(self):
        self.tokenizer, self.model = load_bm1()


    def test_ablate_bm1_layer(self):
        ablate_bm1_layer(self.tokenizer, self.model, 0)
        ablate_bm1_layer(self.tokenizer, self.model, 1)


    def test_ablate_bm1_mlp(self):
        ablate_bm1_mlp(self.tokenizer, self.model, 0)
        ablate_bm1_mlp(self.tokenizer, self.model, 1)


    def test_ablate_bm1_head(self):
        #cfg = qmi.AlgoConfig()
        #cfg.d_vocab = self.model.config.vocab_size
        #cfg.d_model = self.model.config.hidden_size
        #cfg.n_layers = self.model.config.num_layers
        #cfg.n_heads = self.model.config.num_attention_heads
        #cfg.d_head = self.model.config.hidden_size // self.model.config.num_attention_heads
    
        #ablate_bm1_head(cfg, self.tokenizer, self.model, 0, 5)   
        #ablate_bm1_head(cfg, self.tokenizer, self.model, 1, 4)   
        print("TBD")


    def tearDown(self):
        del self.tokenizer
        del self.model