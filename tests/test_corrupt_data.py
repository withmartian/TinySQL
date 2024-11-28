
import unittest
from QuantaTextToSql.ablate import collect_m1_activations, ablated_m1_inference
from QuantaTextToSql.load_data import load_sql_interp_model
from QuantaTextToSql.training_data import (generate_cs1, generate_cs2, generate_cs3, evaluate_cs1_predictions, evaluate_cs2_predictions, 
            evaluate_cs3_predictions, generate_inputs_from_prompt, generate_inputs_from_BatchItems)


class TestCorruptData(unittest.TestCase):

    def setUpModel(self, the_tokenizer, the_model, batch_items, the_inputs, cached_acts):
        self.tokenizer = the_tokenizer
        self.model = the_model
        self.batch_items = batch_items
        self.inputs = the_inputs
        self.cached_acts = cached_acts
        self.max_words = 20

    # Run ablation code with no ablation applied on a tiny story
    def ablate_inference(self, node_type="", layer_index=0, head_index=None):
        return ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, self.inputs, node_type=node_type, layer_index=layer_index, head_index=head_index, max_words=self.max_words)   

    # Run ablation code with no ablation applied on a tiny story
    def ablate_story(self):
        self.inputs = generate_inputs_from_prompt(self.tokenizer, prompt_text="Once upon a time, in a small village, there was a")
        (_, generated_text) = self.ablate_inference()   
        return generated_text
    
    # Run ablation code with no ablation applied 
    def ablate_none(self):
        (_, generated_text) = self.ablate_inference() 
        print(f"No Hook: {generated_text[0]}")
        return generated_text

    def ablate_layer(self):   
        (_, generated_text) = self.ablate_inference(node_type="layer", layer_index=0) 
        (_, generated_text) = self.ablate_inference(node_type="layer", layer_index=1) 
        print(f"Layer Hook: {generated_text[0]}")
        return generated_text

    def ablate_mlp(self):   
        (_, generated_text) = self.ablate_inference(node_type="mlp", layer_index=0) 
        (_, generated_text) = self.ablate_inference(node_type="mlp", layer_index=1) 
        print(f"MLP Hook: {generated_text[0]}")
        return generated_text

    def ablate_head(self):    
        (_, generated_text) = self.ablate_inference(node_type="head", layer_index=0, head_index=5) 
        (_, generated_text) = self.ablate_inference(node_type="head", layer_index=1, head_index=4) 
        print(f"Head Hook: {generated_text[0]}")
        return generated_text

    def tearDown(self):
        del self.tokenizer
        del self.model
        del self.batch_items
        del self.inputs
        del self.cached_acts

