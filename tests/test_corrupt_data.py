import unittest
from QuantaTextToSql.load_data import load_sql_interp_model
from QuantaTextToSql import UNKNOWN_VALUE, ENGTABLENAME, ENGFIELDNAME, DEFTABLESTART, DEFTABLENAME, DEFFIELDSEPARATOR, DEFFIELDNAME, CorruptFeatureTestGenerator
from tests.test_util import TEST_USE_FLASH_ATTENTION,TEST_DEVICE_MAP

class TestCorruptData(unittest.TestCase):

    def show_examples(self, feature_name):
       # TinyStories never uses flash attention
        model_num = 1
        cs_num = 1
        tokenizer, model = load_sql_interp_model(model_num, cs_num, use_flash_attention=False, device_map=TEST_DEVICE_MAP)
         
        generator = CorruptFeatureTestGenerator(model_num=model_num, cs_num=cs_num, tokenizer=tokenizer)

        batch_size = 5
        examples = generator.generate_feature_examples(feature_name, batch_size)      

        for i, example in enumerate(examples, 1):
            print(f"\nExample {i} of {example.corrupted_feature}:")
            if example.corrupted_feature.startswith("Def"):
                print(f"Clean statement: {example.create_statement}")
                print(f"Corrupt statement: {example.corrupted_create_statement}")
            else:
                print(f"Clean prompt: {example.english_prompt}")
                print(f"Corrupt prompt: {example.corrupted_english_prompt}")

            # print("Clean:", example.clean_BatchItem.get_alpaca_prompt())    
            # print("Corrupt:", example.corrupt_BatchItem.get_alpaca_prompt())
            # print("Clean index:", example.clean_token_index)    
            # print("Corrupt index:", example.corrupt_token_index)
            
            assert example.clean_token_index != UNKNOWN_VALUE
            assert example.corrupt_token_index != UNKNOWN_VALUE
            assert example.clean_token_index != example.corrupt_token_index  

        return generator, examples
 
    def test_generate_ENGTABLENAME(self): 
        self.show_examples(ENGTABLENAME)
        
    def test_generate_ENGFIELDNAME(self):   
        self.show_examples(ENGFIELDNAME)
        
    def test_generate_DEFTABLESTART(self):    
        self.show_examples(DEFTABLESTART)

    def test_generate_DEFTABLENAME(self):   
        self.show_examples(DEFTABLENAME)

    def test_generate_DEFFIELDSEPARATOR(self):   
        self.show_examples(DEFFIELDSEPARATOR)

    def test_generate_DEFFIELDNAME(self):  
        self.show_examples(DEFFIELDNAME)
