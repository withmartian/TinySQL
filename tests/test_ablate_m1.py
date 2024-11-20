import unittest
from QuantaTextToSql.ablate import load_bm1, load_bm1_cs1, load_bm1_cs2, load_bm1_cs3, collect_m1_activations, ablated_m1_inference
from QuantaTextToSql.training_data import (generate_cs1, generate_cs2, generate_cs3, evaluate_cs1_predictions, evaluate_cs2_predictions, 
            evaluate_cs3_predictions, generate_inputs_from_prompt, generate_inputs_from_BatchItems)


class TestAblate(unittest.TestCase):

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


# Test the ablation code with the base (unrefined) model
class TestAblate_BM1(TestAblate):

    def setUp(self):
        tokenizer, model = load_bm1()
        batch_items = generate_cs1(25)
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, batch_items) 
        cached_acts = collect_m1_activations(model, inputs)     
        self.setUpModel(tokenizer, model, batch_items, inputs, cached_acts)
 
    def test_story(self):
        generated_text = self.ablate_story()
        expected_text = "Once upon a time, in a small village, there was a little girl named Lucy."
        assert generated_text[0].startswith(expected_text)
    
    def test_code(self):
        
        self.ablate_layer()

        self.ablate_mlp()

        self.ablate_head()

        self.ablate_none()


# Test the ablation code with the model which was refined on CS1
class TestAblate_BM1_CS1(TestAblate):
   
    def setUp(self):
        tokenizer, model = load_bm1_cs1()
        batch_items = generate_cs1(10)
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, batch_items) 
        cached_acts = collect_m1_activations(model, inputs)     
        self.setUpModel(tokenizer, model, batch_items, inputs, cached_acts)
 
    def test_code(self):
        self.ablate_story()
 
        self.ablate_layer()

        self.ablate_mlp()

        self.ablate_head()

    def test_impact(self):
        generated_text = self.ablate_none()
        avg_accuracy1 = evaluate_cs1_predictions(self.batch_items, generated_text)
        print(f"No ablation: Prompt: {self.batch_items[0].get_alpaca_prompt()}")
        print(f"No ablation: Prediction: {avg_accuracy1:4f} {generated_text[0]}")    

        for node_type in ["layer", "mlp"]:

            (_, generated_text) = self.ablate_inference(node_type=node_type, layer_index=0)
            avg_accuracy2 = evaluate_cs1_predictions(self.batch_items, generated_text)
            print(f"Ablate {node_type} 0: {avg_accuracy2:4f}, {generated_text[0]}")

            (_, generated_text) = self.ablate_inference(node_type=node_type, layer_index=1)
            avg_accuracy3 = evaluate_cs1_predictions(self.batch_items, generated_text)
            print(f"Ablate {node_type} 1: {avg_accuracy3:4f}, {generated_text[0]}")

            # This model has accuracy ~0.2 so basically random with/without ablation
            # assert avg_accuracy1 + 0.001 >= avg_accuracy2
            # assert avg_accuracy1 + 0.001 >= avg_accuracy3


# Test the ablation code with the model which was refined on CS2
class TestAblate_BM1_CS2(TestAblate):

    def setUp(self):
        tokenizer, model = load_bm1_cs2()
        batch_items = generate_cs2(10)
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, batch_items)
        cached_acts = collect_m1_activations(model, inputs)     
        self.setUpModel(tokenizer, model, batch_items, inputs, cached_acts)
 
    def test_code(self):
        self.ablate_story()

        self.ablate_layer()

        self.ablate_mlp()

        self.ablate_head()

    def test_impact(self):
        generated_text = self.ablate_none()
        avg_accuracy1 = evaluate_cs2_predictions(self.batch_items, generated_text)
        print(f"No ablation: Prompt: {self.batch_items[0].get_alpaca_prompt()}")
        print(f"No ablation: Prediction: {avg_accuracy1} {generated_text[0]}")        

        for node_type in ["layer", "mlp"]:

            (_, generated_text) = self.ablate_inference( node_type=node_type, layer_index=0)
            avg_accuracy2 = evaluate_cs2_predictions(self.batch_items, generated_text)
            print(f"Ablate {node_type} 0: {avg_accuracy2}, {generated_text[0]}")

            (_, generated_text) = self.ablate_inference( node_type=node_type, layer_index=1)
            avg_accuracy3 = evaluate_cs2_predictions(self.batch_items, generated_text)
            print(f"Ablate {node_type} 1: {avg_accuracy3}, {generated_text[0]}")

            assert avg_accuracy1 + 0.001 >= avg_accuracy2
            assert avg_accuracy1 + 0.001 >= avg_accuracy3

# Test the ablation code with the model which was refined on CS3
class TestAblate_BM1_CS3(TestAblate):

    def setUp(self):
        tokenizer, model = load_bm1_cs3()
        batch_items = generate_cs3(10)
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, batch_items)
        cached_acts = collect_m1_activations(model, inputs)     
        self.setUpModel(tokenizer, model, batch_items, inputs, cached_acts)
 
    def test_code(self):
        self.ablate_story()

        self.ablate_layer()

        self.ablate_mlp()

        self.ablate_head()

    def test_impact(self):
        generated_text = self.ablate_none()
        avg_accuracy1 = evaluate_cs3_predictions(self.batch_items, generated_text)
        print(f"No ablation: Prompt: {self.batch_items[0].get_alpaca_prompt()}")
        print(f"No ablation: Prediction: {avg_accuracy1} {generated_text[0]}")        

        for node_type in ["layer", "mlp"]:

            (_, generated_text) = self.ablate_inference( node_type=node_type, layer_index=0)
            avg_accuracy2 = evaluate_cs3_predictions(self.batch_items, generated_text)
            print(f"Ablate {node_type} 0: {avg_accuracy2}, {generated_text[0]}")

            (_, generated_text) = self.ablate_inference( node_type=node_type, layer_index=1)
            avg_accuracy3 = evaluate_cs3_predictions(self.batch_items, generated_text)
            print(f"Ablate {node_type} 1: {avg_accuracy3}, {generated_text[0]}")

            assert avg_accuracy1 + 0.001 >= avg_accuracy2
            assert avg_accuracy1 + 0.001 >= avg_accuracy3        
