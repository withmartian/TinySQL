import unittest

from tests.test_fragments import TestFragments
from tests.test_cs1 import TestCommandSet1
from tests.test_cs2 import TestCommandSet2
from tests.test_cs3 import TestCommandSet3
from tests.test_ablate_m1 import TestAblate_BM1, TestAblate_BM1_CS1, TestAblate_BM1_CS2, TestAblate_BM1_CS3
from tests.test_ablate_m2 import TestAblate_BM2


if __name__ == '__main__':
    test_classes_to_run = [
        TestFragments,
        TestCommandSet1,
        TestCommandSet2,
        TestCommandSet3,
        TestAblate_BM1,
        TestAblate_BM1_CS1,
        TestAblate_BM1_CS2,
        TestAblate_BM1_CS3]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
