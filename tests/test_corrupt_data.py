import unittest
from TinySQL.load_data import load_sql_interp_model
from TinySQL import UNKNOWN_VALUE, ENGTABLENAME, ENGFIELDNAME, DEFTABLENAME, DEFFIELDNAME, CorruptFeatureTestGenerator
from tests.test_util import TEST_DEVICE_MAP

class TestCorruptData(unittest.TestCase):

    def show_examples(self, feature_name, model_num, cs_num=1, use_novel_names=False, use_synonyms_field=False, use_synonyms_table=False):
        tokenizer, _ = load_sql_interp_model(model_num, cs_num, use_flash_attention=False, device_map=TEST_DEVICE_MAP)
                 
        generator = CorruptFeatureTestGenerator(
            model_num=model_num, 
            cs_num=cs_num, 
            tokenizer=tokenizer, 
            use_novel_names=use_novel_names, 
            use_synonyms_field=use_synonyms_field,
            use_synonyms_table=use_synonyms_table
        )

        batch_size = 5
        examples = generator.generate_feature_examples(feature_name, batch_size)      

        len_all_tokens = len(tokenizer(examples[0].clean_BatchItem.get_alpaca_prompt() + examples[0].clean_BatchItem.sql_statement)["input_ids"])

        for i, example in enumerate(examples, 1):
            print(f"\nExample {i} of {example.feature_name}: cs_num={cs_num} use_novel_names={use_novel_names} use_synonyms_field={use_synonyms_field} use_synonyms_table={use_synonyms_table} ")
            print(f"cleantablenames=({example.clean_BatchItem.table_name.name},{example.clean_BatchItem.table_name.synonym})")

            if example.feature_name.startswith("Def"):
                print(f"Clean statement  : {example.clean_token_str} : {example.create_statement}")
                print(f"Corrupt statement: {example.corrupt_token_str} : {example.corrupt_create_statement}")
            else:
                print(f"Clean prompt  : {example.clean_token_str} : {example.english_prompt}")
                print(f"Corrupt prompt: {example.corrupt_token_str} : {example.corrupt_english_prompt}")

            assert example.clean_token_str != ""
            assert example.corrupt_token_str != ""
            assert example.clean_token_str != example.corrupt_token_str
            assert example.clean_tokenizer_index != UNKNOWN_VALUE
            assert example.corrupt_tokenizer_index != UNKNOWN_VALUE
            assert example.clean_tokenizer_index != example.corrupt_tokenizer_index  
            assert example.prompt_token_index > 0
            assert example.answer_token_index > 0
            assert example.prompt_token_index < example.answer_token_index 

            cleanBatchItem = example.clean_BatchItem
            corruptBatchItem = example.corrupt_BatchItem
            assert cleanBatchItem.command_set == corruptBatchItem.command_set
            assert cleanBatchItem.table_name.use_synonym == corruptBatchItem.table_name.use_synonym
            assert cleanBatchItem.table_name.name == corruptBatchItem.table_name.name
            assert cleanBatchItem.table_name.synonym == corruptBatchItem.table_name.synonym

            clean_str = cleanBatchItem.get_alpaca_prompt() + example.clean_BatchItem.sql_statement
            corrupt_str = corruptBatchItem.get_alpaca_prompt() + example.corrupt_BatchItem.sql_statement
            if clean_str == corrupt_str:
                print("Clean  :", clean_str)
                print("Corrupt:", corrupt_str)
                assert False

            clean_tokens = tokenizer(clean_str)["input_ids"]
            corrupt_tokens = tokenizer(corrupt_str)["input_ids"]
            if clean_tokens == corrupt_tokens:
                print(clean_tokens)
                print(corrupt_tokens)
                assert False

            if example.clean_tokenizer_index == example.corrupt_tokenizer_index:
                print(example.clean_tokenizer_index)
                print(example.corrupt_tokenizer_index)
                assert False

            # All (clean or corrupt) examples should have the same prompt+answer length
            assert len(clean_tokens) == len_all_tokens
            assert len(corrupt_tokens) == len_all_tokens

            # Check the clean string tokens have expected clean token in prompt and answer. 
            print(f"Prompt token index: {example.prompt_token_index} {example.answer_token_index} {len(clean_tokens)} {example.clean_tokenizer_index} ")
            assert clean_tokens[example.prompt_token_index] == example.clean_tokenizer_index
            # If we are testing EngTableFeature or EngFieldFeature, with a semantic model, clean token will not be in the answer
            if example.feature_name.startswith("Def"):
                assert clean_tokens[example.answer_token_index] == example.clean_tokenizer_index

            # Check the corrupt string tokens have expected corrupt token in prompt 
            # PQR TODO This does not work for modelNum==2. Not sure why
            if model_num == 1: 
                prompt_corrupt_token = corrupt_tokens[example.prompt_token_index]
                if prompt_corrupt_token != example.corrupt_tokenizer_index:
                    print("Bad prompt corrupt token[", example.prompt_token_index, "]")
                    print("Clean  :", example.clean_token_str, " - ", clean_str)
                    print("Corrupt:", example.corrupt_token_str, " - ", corrupt_str)
                    print("Want:", tokenizer.decode(example.corrupt_tokenizer_index)) 
                    print("Found:", tokenizer.decode(prompt_corrupt_token)) 
                    assert False

        return generator, examples

    def check_word_is_one_token(self, prefix, tokenizer, word):
        if len(tokenizer(word)["input_ids"]) != 1:
            print(prefix, "Word not 1 token:", word)
            assert False

    def check_words_are_one_token(self, prefix, tokenizer, words):
        for word in words:
            self.check_word_is_one_token(prefix, tokenizer, word)

    # Check that all the clean and corrupt tokens are single tokens
    # So that when we generate the paired clean and corrupt examples, they have the same number of tokens  
    def test_m1_generate_name_tokens(self): 
        tokenizer, _ = load_sql_interp_model(1, 2, use_flash_attention=False, device_map=TEST_DEVICE_MAP)
         
        generator = CorruptFeatureTestGenerator(model_num=1, cs_num=2, tokenizer=tokenizer, use_novel_names=True)

        self.check_words_are_one_token("clean_table_name", tokenizer, generator.clean_table_names)   
        self.check_words_are_one_token("novel_table_name", tokenizer, generator.novel_table_names)
        self.check_words_are_one_token("synonym_table_name.keys", tokenizer, generator.synonym_table_names.keys())
        self.check_words_are_one_token("clean_field_name", tokenizer, generator.clean_field_names)
        self.check_words_are_one_token("novel_field_name", tokenizer, generator.novel_field_names)
        self.check_words_are_one_token("synonym_field_name.keys", tokenizer, generator.synonym_field_names.keys())
        self.check_words_are_one_token("clean_field_type", tokenizer, generator.clean_field_types)

        # These are novel words, so they are not in the tokenizer, and will not be one token
        #for key in generator.synonym_table_names.keys():
        #    self.check_word_is_one_token("synonym_table_name[key]", tokenizer, generator.synonym_table_names[key])
        #for key in generator.synonym_field_names.keys():
        #    self.check_word_is_one_token("synonym_field_name[key]", tokenizer, generator.synonym_field_names[key])


    def test_m1_generate_ENGTABLENAME(self): 
        self.show_examples(ENGTABLENAME, 1, use_novel_names=False, use_synonyms_table=True) # This fails
        self.show_examples(ENGTABLENAME, 1, use_novel_names=True, use_synonyms_table=True)
        self.show_examples(ENGTABLENAME, 1, use_novel_names=False)
        self.show_examples(ENGTABLENAME, 1, use_novel_names=True)

    def test_m1_generate_ENGFIELDNAME(self):   
        self.show_examples(ENGFIELDNAME, 1, use_novel_names=False)
        self.show_examples(ENGFIELDNAME, 1, use_novel_names=True)
        self.show_examples(ENGFIELDNAME, 1, use_novel_names=False, use_synonyms_field=True)
        self.show_examples(ENGFIELDNAME, 1, use_novel_names=True, use_synonyms_field=True)

    def test_m1_generate_DEFTABLENAME(self):   
        self.show_examples(DEFTABLENAME, 1, use_novel_names=False)
        self.show_examples(DEFTABLENAME, 1, use_novel_names=True)
        # We are changing the table name in the Instructions so we do not vary the table name in the Context using synonyms.  
        #self.show_examples(DEFTABLENAME, 1, use_novel_names=False, use_synonyms_table=True)
        #self.show_examples(DEFTABLENAME, 1, use_novel_names=True, use_synonyms_table=True)

    def test_m1_generate_DEFFIELDNAME(self):  
        self.show_examples(DEFFIELDNAME, 1, use_novel_names=False)
        self.show_examples(DEFFIELDNAME, 1, use_novel_names=True)
        self.show_examples(DEFFIELDNAME, 1, use_novel_names=False, use_synonyms_field=True)
        self.show_examples(DEFFIELDNAME, 1, use_novel_names=True, use_synonyms_field=True)

    def test_m2_generate_DEFFIELDNAME(self):  
        self.show_examples(DEFFIELDNAME, 2, use_novel_names=False)
        self.show_examples(DEFFIELDNAME, 2, use_novel_names=True)
        self.show_examples(DEFFIELDNAME, 2, use_novel_names=False, use_synonyms_field=True)
        self.show_examples(DEFFIELDNAME, 2, use_novel_names=True, use_synonyms_field=True)

    # Test that CorruptFeatureTestGenerator generates examples with the same number of tokens   
    # under a range of different conditions. 
    def test_m1_generate_same_length(self):   
        tokenizer, _ = load_sql_interp_model(1, 2, use_flash_attention=False, device_map=TEST_DEVICE_MAP)

        batch_size = 10

        for use_novel_names in [True,False]:
            for use_synonyms_field in [True,False]:
                for use_synonyms_table in [True,False]:
                    for feature_name in [ENGTABLENAME, ENGFIELDNAME, DEFTABLENAME, DEFFIELDNAME]:

                        if feature_name == DEFTABLENAME and use_synonyms_table:
                            continue

                        generator = CorruptFeatureTestGenerator(
                            model_num=1, 
                            cs_num=2, 
                            tokenizer=tokenizer, 
                            use_novel_names=use_novel_names, 
                            use_synonyms_field=use_synonyms_field,
                            use_synonyms_table=use_synonyms_table)

                        examples = generator.generate_feature_examples(feature_name, batch_size)      

                        tokens = len(tokenizer(examples[0].clean_BatchItem.get_alpaca_prompt() + examples[0].clean_BatchItem.sql_statement)["input_ids"])
                        print(f"{feature_name} use_novel_names={use_novel_names} use_synonyms_field={use_synonyms_field} use_synonyms_table={use_synonyms_table} tokens={tokens}")

                        for i, example in enumerate(examples, 1):

                            this_tokens = len(tokenizer(example.clean_BatchItem.get_alpaca_prompt() + example.clean_BatchItem.sql_statement)["input_ids"])
                            assert this_tokens == tokens

                            this_tokens = len(tokenizer(example.corrupt_BatchItem.get_alpaca_prompt() + example.corrupt_BatchItem.sql_statement)["input_ids"])
                            assert this_tokens == tokens
