import unittest
from QuantaTextToSql.ablate import load_bm1, load_bm1_cs1, load_bm1_cs2, load_bm1_cs3, collect_m1_activations, ablated_m1_inference
from QuantaTextToSql.training_data import generate_cs1, generate_cs2, generate_cs3
from QuantaTextToSql.training_data.generate_utils import generate_inputs_from_prompt, generate_inputs_from_BatchItems
import QuantaMechInterp as qmi


class TestAblate(unittest.TestCase):

    def setUpModel(self, the_tokenizer, the_model, cached_acts):
        self.tokenizer = the_tokenizer
        self.model = the_model
        self.cached_acts = cached_acts
 
    # Run ablation code with no ablation applied on a tiny story
    def ablate_bm1_story(self):
        inputs = generate_inputs_from_prompt(self.tokenizer, prompt_text="Once upon a time, in a small village, there was a")
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="", layer_index=0)   
        print(f"Tiny Story: {generated_text[0]}") # e.g. "little"
    
    # Run ablation code with no ablation applied 
    def ablate_bm1_none(self, batch_items):
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="", layer_index=0)
        print(f"No Hook: {generated_text[0]}")

    def ablate_bm1_layer(self, batch_items):
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="layer", layer_index=0)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="layer", layer_index=1)
        print(f"Layer Hook: {generated_text[0]}")

    def ablate_bm1_mlp(self, batch_items):
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="mlp", layer_index=0)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="mlp", layer_index=1)
        print(f"MLP Hook: {generated_text[0]}")

    def ablate_bm1_head(self, batch_items):
        (_, inputs) = generate_inputs_from_BatchItems(self.tokenizer, batch_items)        
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="head", layer_index=0, head_index=5)
        (_, generated_text) = ablated_m1_inference(self.tokenizer, self.model, self.cached_acts, inputs, node_type="head", layer_index=1, head_index=4)
        print(f"Head Hook: {generated_text[0]}")

    def tearDownModel(self):
        del self.tokenizer
        del self.model
        del self.cached_acts


# Test the ablation code with the base (unrefined) model
class TestAblateBM1(TestAblate):

    def setUp(self):
        tokenizer, model = load_bm1()
        batch_items = generate_cs1(100)
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, batch_items)
        cached_acts = collect_m1_activations(model, inputs)     
        self.setUpModel(tokenizer, model, cached_acts)
 
    def test_ablate_bm1_story(self):
        self.ablate_bm1_story()
    
    def test_ablate_bm1_none(self):
        batch_items = generate_cs1(100)
        self.ablate_bm1_none(batch_items)

    def test_ablate_bm1_layer(self):
        batch_items = generate_cs1(100)
        self.ablate_bm1_layer(batch_items)

    def test_ablate_bm1_mlp(self):
        batch_items = generate_cs1(100)
        self.ablate_bm1_mlp(batch_items)

    def test_ablate_bm1_head(self):
        batch_items = generate_cs1(100)
        self.ablate_bm1_head(batch_items)

    def tearDown(self):
        self.tearDownModel()


# Test the ablation code with the model which was refined on CS1
class TestAblate_BM1_CS1(TestAblate):

    def generate_batch_items(self):
        return generate_cs1(100)
    
    def setUp(self):
        tokenizer, model = load_bm1_cs1()
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, self.generate_batch_items())
        cached_acts = collect_m1_activations(model, inputs)     
        self.setUpModel(tokenizer, model, cached_acts)
 
    def test_ablate_bm1_cs1_story(self):
        self.ablate_bm1_story()

    def test_ablate_bm1_cs1_none(self):
        self.ablate_bm1_none(self.generate_batch_items())

    def test_ablate_bm1_cs1_layer(self):
        self.ablate_bm1_layer(self.generate_batch_items())

    def test_ablate_bm1_cs1_mlp(self):
        self.ablate_bm1_mlp(self.generate_batch_items())

    def test_ablate_bm1_cs1_head(self):
        self.ablate_bm1_head(self.generate_batch_items())

    def tearDown(self):
        self.tearDownModel()


# Test the ablation code with the model which was refined on CS2
class TestAblate_BM1_CS2(TestAblate):

    def generate_batch_items(self):
        return generate_cs2(100)
    
    def setUp(self):
        tokenizer, model = load_bm1_cs2()
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, self.generate_batch_items())
        cached_acts = collect_m1_activations(model, inputs)     
        self.setUpModel(tokenizer, model, cached_acts)
 
    def test_ablate_bm1_cs2_story(self):
        self.ablate_bm1_story()

    def test_ablate_bm1_cs2_none(self):
        self.ablate_bm1_none(self.generate_batch_items())

    def test_ablate_bm1_cs2_layer(self):
        self.ablate_bm1_layer(self.generate_batch_items())

    def test_ablate_bm1_cs2_mlp(self):
        self.ablate_bm1_mlp(self.generate_batch_items())

    def test_ablate_bm1_cs2_head(self):
        self.ablate_bm1_head(self.generate_batch_items())

    def tearDown(self):
        self.tearDownModel()


# Test the ablation code with the model which was refined on CS3
class TestAblate_BM1_CS3(TestAblate):

    def generate_batch_items(self):
        return generate_cs3(100)

    def setUp(self):
        tokenizer, model = load_bm1_cs3()
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, self.generate_batch_items())
        cached_acts = collect_m1_activations(model, inputs)     
        self.setUpModel(tokenizer, model, cached_acts)
 
    def test_ablate_bm1_cs3_story(self):
        self.ablate_bm1_story()

    def test_ablate_bm1_cs3_none(self):
        self.ablate_bm1_none(self.generate_batch_items())

    def test_ablate_bm1_cs3_layer(self):
        self.ablate_bm1_layer(self.generate_batch_items())

    def test_ablate_bm1_cs3_mlp(self):
        self.ablate_bm1_mlp(self.generate_batch_items())

    def test_ablate_bm1_cs3_head(self):
        self.ablate_bm1_head(self.generate_batch_items())

    def tearDown(self):
        self.tearDownModel()
