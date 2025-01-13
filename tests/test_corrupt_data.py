import unittest
from TinySQL.load_data import load_sql_interp_model
from TinySQL import UNKNOWN_VALUE, ENGTABLENAME, ENGFIELDNAME, DEFCREATETABLE, DEFTABLENAME, DEFFIELDSEPARATOR, DEFFIELDNAME, CorruptFeatureTestGenerator
from tests.test_util import TEST_DEVICE_MAP

class TestCorruptData(unittest.TestCase):

    def show_examples(self, feature_name, model_num, cs_num=1, use_novel_names=False):
        tokenizer, _ = load_sql_interp_model(model_num, cs_num, use_flash_attention=False, device_map=TEST_DEVICE_MAP)
         
        generator = CorruptFeatureTestGenerator(model_num=model_num, cs_num=cs_num, tokenizer=tokenizer, use_novel_names=use_novel_names)

        batch_size = 5
        examples = generator.generate_feature_examples(feature_name, batch_size)      

        len_all_tokens = len(tokenizer(examples[0].clean_BatchItem.get_alpaca_prompt() + examples[0].clean_BatchItem.sql_statement)["input_ids"])

        for i, example in enumerate(examples, 1):
            print(f"\nExample {i} of {example.feature_name}:")
            if example.feature_name.startswith("Def"):
                print(f"Clean statement  : {example.create_statement}")
                print(f"Corrupt statement: {example.corrupt_create_statement}")
            else:
                print(f"Clean prompt  : {example.english_prompt}")
                print(f"Corrupt prompt: {example.corrupt_english_prompt}")

            assert example.clean_token_str != ""
            assert example.corrupt_token_str != ""
            assert example.clean_token_str != example.corrupt_token_str
            assert example.clean_tokenizer_index != UNKNOWN_VALUE
            assert example.corrupt_tokenizer_index != UNKNOWN_VALUE
            assert example.clean_tokenizer_index != example.corrupt_tokenizer_index  
            assert example.prompt_token_index > 0
            assert example.answer_token_index > 0
            assert example.prompt_token_index < example.answer_token_index 

            clean_str = example.clean_BatchItem.get_alpaca_prompt() + example.clean_BatchItem.sql_statement
            corrupt_str = example.corrupt_BatchItem.get_alpaca_prompt() + example.corrupt_BatchItem.sql_statement
            clean_tokens = tokenizer(clean_str)["input_ids"]
            corrupt_tokens = tokenizer(corrupt_str)["input_ids"]

            # All (clean or corrupt) examples should have the same prompt+answer length
            assert len(clean_tokens) == len_all_tokens
            assert len(corrupt_tokens) == len_all_tokens

            print(f"Prompt token index: {example.prompt_token_index} {example.answer_token_index} {example.clean_tokenizer_index} {len(clean_tokens)}")
            assert clean_tokens[example.prompt_token_index] == example.clean_tokenizer_index
            assert clean_tokens[example.answer_token_index] == example.clean_tokenizer_index

            # PQR TODO This does not work for modelNum==2. Not sure why
            if model_num <= 1: 
                if corrupt_tokens[example.prompt_token_index] != example.corrupt_tokenizer_index:
                    print(clean_str)
                    print(corrupt_str)
                    print( "Bad prompt corrupt token:", example.prompt_token_index, example.corrupt_tokenizer_index, corrupt_tokens[example.prompt_token_index], corrupt_tokens)
                    assert False
                #if corrupt_tokens[example.answer_token_index] != example.corrupt_tokenizer_index:
                #    print("Bad answer corrupt token:", example.answer_token_index, example.corrupt_tokenizer_index, corrupt_tokens[example.answer_token_index], corrupt_tokens)
                #    assert False

        return generator, examples
 

    # Check that all the clean and corrupt tokens are single tokens
    # So that when we generate the paired clean and corrupt examples, they have the same number of tokens  
    def test_m1_generate_name_tokens(self): 
        tokenizer, _ = load_sql_interp_model(1, 2, use_flash_attention=False, device_map=TEST_DEVICE_MAP)
         
        generator = CorruptFeatureTestGenerator(model_num=1, cs_num=2, tokenizer=tokenizer, use_novel_names=True)

        for word in generator.clean_table_names:
            assert len(tokenizer(word)["input_ids"]) == 1

        for word in generator.novel_table_names:
            assert len(tokenizer(word)["input_ids"]) == 1

        for word in generator.clean_field_names:
            assert len(tokenizer(word)["input_ids"]) == 1

        for word in generator.novel_field_names:
            assert len(tokenizer(word)["input_ids"]) == 1

        for word in generator.clean_field_types:
            assert len(tokenizer(word)["input_ids"]) == 1


    def test_m1_generate_ENGTABLENAME(self): 
        self.show_examples(ENGTABLENAME, 1, use_novel_names=False)
        self.show_examples(ENGTABLENAME, 1, use_novel_names=True)


    def test_m1_generate_ENGFIELDNAME(self):   
        self.show_examples(ENGFIELDNAME, 1, use_novel_names=False)
        self.show_examples(ENGFIELDNAME, 1, use_novel_names=True)

    # Suppress until CREATE is in the TinyStories Vocab     
    #def test_generate_DEFCREATETABLE(self):    
    #    self.show_examples(DEFCREATETABLE, 1)

    def test_m1_generate_DEFTABLENAME(self):   
        self.show_examples(DEFTABLENAME, 1, use_novel_names=False)
        self.show_examples(DEFTABLENAME, 1, use_novel_names=True)

    # Need to debug how "," is tokenized 
    #def test_generate_DEFFIELDSEPARATOR(self):   
    #    self.show_examples(DEFFIELDSEPARATOR, 1)

    def test_m1_generate_DEFFIELDNAME(self):  
        self.show_examples(DEFFIELDNAME, 1, use_novel_names=False)
        self.show_examples(DEFFIELDNAME, 1, use_novel_names=True)

    def test_m2_generate_DEFFIELDNAME(self):  
        self.show_examples(DEFFIELDNAME, 2, use_novel_names=False)
        self.show_examples(DEFFIELDNAME, 2, use_novel_names=True)

