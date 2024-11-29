
import unittest
from QuantaTextToSql import FEATURE_TESTS


class TestCorruptData(unittest.TestCase):

    def test_generate_corrupt_data(self):
        # Test all features
        for test in FEATURE_TESTS:
            print(f"\nTesting feature: {test.corrupted_feature}")
            
            if test.corrupted_create_statement:
                print(f"Clean create: {test.create_statement}")
                print(f"Corrupt create: {test.corrupted_create_statement}")
            elif test.corrupted_english_prompt:
                print(f"Clean prompt: {test.english_prompt}")
                print(f"Corrupt prompt: {test.corrupted_english_prompt}")
  
