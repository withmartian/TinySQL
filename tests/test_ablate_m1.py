import unittest
from QuantaTextToSql.ablate import load_bm1, load_bm1_cs1, load_bm1_cs2, load_bm1_cs3, collect_m1_activations, ablated_m1_inference
from QuantaTextToSql.training_data import generate_cs1
from QuantaTextToSql.training_data.generate_utils import generate_inputs_from_prompt, generate_inputs_from_BatchItems
import QuantaMechInterp as qmi


class TestAblateBM1(unittest.TestCase):

    def setUp(self):
        self.tokenizer, self.model = load_bm1()

        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)
        self.cached_acts = collect_m1_activations(self.model, inputs)
        #print(f"Max Length: Prompts {len(prompts[0])}, Inputs {len(inputs[0])}")
 
    # Run ablation code with no ablation applied on a tiny story
    def test_ablate_bm1_story(self):
        inputs = generate_inputs_from_prompt(self.tokenizer, prompt_text="Once upon a time, in a small village, there was a")
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="", layer_index=0)   
        print(f"Tiny Story: {generated_text}") # e.g. "little"
    
    # Run ablation code with no ablation applied 
    def test_ablate_bm1_none(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="", layer_index=0)
        print(f"No Hook: {generated_text}")

    def test_ablate_bm1_layer(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="layer", layer_index=0)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="layer", layer_index=1)
        print(f"Layer Hook: {generated_text}")

    def test_ablate_bm1_mlp(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="mlp", layer_index=0)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="mlp", layer_index=1)
        print(f"MLP Hook: {generated_text}")

    def test_ablate_bm1_head(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="head", layer_index=0, head_index=5)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="head", layer_index=1, head_index=4)
        print(f"Head Hook: {generated_text}")

    def tearDown(self):
        del self.tokenizer
        del self.model


class TestAblate_BM1_CS1(unittest.TestCase):

    def setUp(self):
        self.tokenizer, self.model = load_bm1_cs1()

        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)
        self.cached_acts = collect_m1_activations(self.model, inputs)
        #print(f"Max Length: Prompts {len(prompts[0])}, Inputs {len(inputs[0])}")
 
    # Run ablation code with no ablation applied on a tiny story
    def test_ablate_bm1_cs1_story(self):
        inputs = generate_inputs_from_prompt(self.tokenizer, prompt_text="Once upon a time, in a small village, there was a")
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="", layer_index=0)   
        print(f"Tiny Story: {generated_text}") # e.g. "little"

    def test_ablate_bm1_cs1_none(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="", layer_index=0)
        print(f"No Hook: {generated_text}")

    def test_ablate_bm1_cs1_layer(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="layer", layer_index=0)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="layer", layer_index=1)
        print(f"Layer Hook: {generated_text}")

    def test_ablate_bm1_cs1_mlp(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="mlp", layer_index=0)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="mlp", layer_index=1)
        print(f"MLP Hook: {generated_text}")

    def test_ablate_bm1_cs1_head(self):
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="head", layer_index=0, head_index=5)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="head", layer_index=1, head_index=4)
        print(f"Head Hook: {generated_text}")

    def tearDown(self):
        del self.tokenizer
        del self.model
