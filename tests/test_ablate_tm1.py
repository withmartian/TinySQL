import unittest
from QuantaTextToSql.ablate import load_tm1, collect_tm1_activations, ablate_tm1
from QuantaTextToSql.training_data import generate_cs1
from QuantaTextToSql.training_data.generate_utils import generate_inputs_from_BatchItems
import QuantaMechInterp as qmi


class TestAblateTM1(unittest.TestCase):

    def setUp(self):
        self.tokenizer, self.model = load_tm1()

        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)
        collect_tm1_activations(self.model, self.tokenizer, inputs)
        #print(f"Max Length: Prompts {len(prompts[0])}, Inputs {len(inputs[0])}")
 
    def test_ablate_tm1_none(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)
        (_, generated_text) = ablate_tm1(self.tokenizer, self.model, inputs, node_type="", layer_index=0)
        print(f"No Hook: {generated_text[0]}")

    def test_ablate_tm1_layer(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablate_tm1(self.tokenizer, self.model, inputs, node_type="layer", layer_index=0)
        (_, generated_text) = ablate_tm1(self.tokenizer, self.model, inputs, node_type="layer", layer_index=1)
        print(f"Layer Hook: {generated_text[0]}")

    def test_ablate_tm1_mlp(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablate_tm1(self.tokenizer, self.model, inputs, node_type="mlp", layer_index=0)
        (_, generated_text) = ablate_tm1(self.tokenizer, self.model, inputs, node_type="mlp", layer_index=1)
        print(f"MLP Hook: {generated_text[0]}")

    def test_ablate_tm1_head(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablate_tm1(self.tokenizer, self.model, inputs, node_type="head", layer_index=0, head_index=5)
        (_, generated_text) = ablate_tm1(self.tokenizer, self.model, inputs, node_type="head", layer_index=1, head_index=4)
        print(f"Head Hook: {generated_text[0]}")

    def tearDown(self):
        del self.tokenizer
        del self.model
