import unittest

from TinySQL.load_data import sql_interp_model_location


class TestLoad(unittest.TestCase):

    def test_sql_interp_model_location(self):
        
        sql_interp_model_location(0, 0)
        sql_interp_model_location(1, 0, True)
        sql_interp_model_location(1, 0, False)
        sql_interp_model_location(1, 1, True)
        sql_interp_model_location(1, 1, False)
        sql_interp_model_location(1, 2, True)
        sql_interp_model_location(1, 2, False)
       