import unittest

from tests.test_cs1 import TestCommandSet1
from tests.test_cs2 import TestCommandSet2


if __name__ == '__main__':
    test_classes_to_run = [TestCommandSet1,TestCommandSet2]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
