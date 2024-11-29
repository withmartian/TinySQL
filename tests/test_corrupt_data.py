
import unittest
from QuantaTextToSql import ENGTABLENAME, ENGFIELDNAME, SQLTABLESTART, SQLTABLENAME, SQLFIELDSEPARATOR, FeatureTestGenerator

class TestCorruptData(unittest.TestCase):

    def show_examples(self, examples):
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i} of {example.corrupted_feature}:")
            if example.corrupted_feature.startswith("Sql"):
                print(f"Clean statement: {example.create_statement}")
                print(f"Corrupt statement: {example.corrupted_create_statement}")
            else:
                print(f"Clean prompt: {example.english_prompt}")
                print(f"Corrupt prompt: {example.corrupted_english_prompt}")

    def test_generate_corrupt_data(self):
        generator = FeatureTestGenerator()
        
        examples = generator.generate_feature_examples(ENGTABLENAME, 2)      
        self.show_examples(examples)
        
        examples = generator.generate_feature_examples(ENGFIELDNAME, 2)      
        self.show_examples(examples)
        
        examples = generator.generate_feature_examples(SQLTABLESTART, 2)      
        self.show_examples(examples)

        examples = generator.generate_feature_examples(SQLTABLENAME, 2)      
        self.show_examples(examples)

        examples = generator.generate_feature_examples(SQLFIELDSEPARATOR, 2)      
        self.show_examples(examples)
