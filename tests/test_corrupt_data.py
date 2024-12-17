import unittest
from TinySQL.load_data import load_sql_interp_model
from TinySQL import UNKNOWN_VALUE, ENGTABLENAME, ENGFIELDNAME, DEFCREATETABLE, DEFTABLENAME, DEFFIELDSEPARATOR, DEFFIELDNAME, CorruptFeatureTestGenerator
from tests.test_util import TEST_DEVICE_MAP

class TestCorruptData(unittest.TestCase):

    def show_examples(self, feature_name):
       # TinyStories never uses flash attention
        model_num = 1
        cs_num = 1
        tokenizer, _ = load_sql_interp_model(model_num, cs_num, use_flash_attention=False, device_map=TEST_DEVICE_MAP)
         
        generator = CorruptFeatureTestGenerator(model_num=model_num, cs_num=cs_num, tokenizer=tokenizer)

        batch_size = 5
        examples = generator.generate_feature_examples(feature_name, batch_size)      

        for i, example in enumerate(examples, 1):
            print(f"\nExample {i} of {example.feature_name}:")
            if example.feature_name.startswith("Def"):
                print(f"Clean statement: {example.create_statement}")
                print(f"Corrupt statement: {example.corrupt_create_statement}")
            else:
                print(f"Clean prompt: {example.english_prompt}")
                print(f"Corrupt prompt: {example.corrupt_english_prompt}")

            if False:
                example.print_clean()
                example.print_corrupt()

            assert example.clean_token != ""
            assert example.corrupt_token != ""
            assert example.clean_token != example.corrupt_token
            assert example.clean_tokenizer_index != UNKNOWN_VALUE
            assert example.corrupt_tokenizer_index != UNKNOWN_VALUE
            assert example.clean_tokenizer_index != example.corrupt_tokenizer_index  
            assert example.prompt_token_index > 0
            assert example.answer_token_index > 0
            assert example.prompt_token_index < example.answer_token_index 

            clean_tokens = tokenizer(example.clean_BatchItem.get_alpaca_prompt() + example.clean_BatchItem.sql_statement)["input_ids"]
            assert clean_tokens[example.prompt_token_index] == example.clean_tokenizer_index
            assert clean_tokens[example.answer_token_index] == example.clean_tokenizer_index

            corrupt_tokens = tokenizer(example.corrupt_BatchItem.get_alpaca_prompt() + example.corrupt_BatchItem.sql_statement)["input_ids"]
            assert corrupt_tokens[example.prompt_token_index] == example.corrupt_tokenizer_index
            #assert corrupt_tokens[example.answer_token_index] == example.corrupt_tokenizer_index      corrupt_BatchItem.sql_statement is not in corrupt_tokens

        return generator, examples
 
    def test_generate_ENGTABLENAME(self): 
        self.show_examples(ENGTABLENAME)
        
    def test_generate_ENGFIELDNAME(self):   
        self.show_examples(ENGFIELDNAME)

    # Suppress until CREATE is in the TinyStories Vocab     
    #def test_generate_DEFCREATETABLE(self):    
    #    self.show_examples(DEFCREATETABLE)

    def test_generate_DEFTABLENAME(self):   
        self.show_examples(DEFTABLENAME)

    # Need to debug how "," is tokenized 
    #def test_generate_DEFFIELDSEPARATOR(self):   
    #    self.show_examples(DEFFIELDSEPARATOR)

    def test_generate_DEFFIELDNAME(self):  
        self.show_examples(DEFFIELDNAME)
