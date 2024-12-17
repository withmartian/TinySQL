from TinySQL.load_data import load_sql_interp_model
from TinySQL.training_data import (generate_cs1, generate_inputs_from_BatchItems)
from tests.test_ablate_m1 import TestAblate
from tests.test_util import TEST_USE_FLASH_ATTENTION,TEST_DEVICE_MAP


# Test the ablation code with the base (unrefined) model
class TestAblate_BM2(TestAblate):

    def setUp(self):
        tokenizer, model = load_sql_interp_model(2, 0, use_flash_attention=TEST_USE_FLASH_ATTENTION, device_map=TEST_DEVICE_MAP)
        batch_items = generate_cs1(25)
        (_, inputs) = generate_inputs_from_BatchItems(tokenizer, batch_items) 
        cached_acts = None #collect_m2_activations(model, inputs)     
        self.setUpModel(tokenizer, model, batch_items, inputs, cached_acts)
 
    def test_story(self):
        # Need to use nnsight ablation on M2: 
        # generated_text = self.ablate_story()
        # expected_text = "Once upon a time, in a small village, there was a little girl named Lucy."
        # assert generated_text[0].startswith(expected_text)
        pass
    
