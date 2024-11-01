import unittest
import sys
sys.path.append('/mnt/foundation-shared/dhruv_gretel_ai/research/sql/quanta_text_to_sql')
from tests.test_cs1_training_data import TestFragments


if __name__ == '__main__':
    test_classes_to_run = [TestFragments]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
