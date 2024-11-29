
import unittest
from QuantaTextToSql import ENGTABLENAME, ENGFIELDNAME, DEFTABLESTART, DEFTABLENAME, DEFFIELDSEPARATOR, DEFFIELDNAME, CorruptFeatureTestGenerator

class TestCorruptData(unittest.TestCase):

    def show_examples(self, case):
        generator = CorruptFeatureTestGenerator()
        examples = generator.generate_feature_examples(case, 2)      

        for i, example in enumerate(examples, 1):
            print(f"\nExample {i} of {example.corrupted_feature}:")
            if example.corrupted_feature.startswith("Def"):
                print(f"Clean statement: {example.create_statement}")
                print(f"Corrupt statement: {example.corrupted_create_statement}")
            else:
                print(f"Clean prompt: {example.english_prompt}")
                print(f"Corrupt prompt: {example.corrupted_english_prompt}")

            #if i == 2:
            #    print("Clean:", example.clean_BatchItem.get_alpaca_prompt())    
            #    print("Corrupt:", example.corrupt_BatchItem.get_alpaca_prompt())    
 
    def test_generate_ENGTABLENAME(self): 
        self.show_examples(ENGTABLENAME)
        
    def test_generate_ENGFIELDNAME(self):   
        self.show_examples(ENGFIELDNAME)
        
    def test_generate_SQLTABLESTART(self):    
        self.show_examples(DEFTABLESTART)

    def test_generate_SQLTABLENAME(self):   
        self.show_examples(DEFTABLENAME)

    def test_generate_SQLFIELDSEPARATOR(self):   
        self.show_examples(DEFFIELDSEPARATOR)

    def test_generate_SQLFIELDNAME(self):  
        self.show_examples(DEFFIELDNAME)
