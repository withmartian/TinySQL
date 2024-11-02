import unittest

from training_data import ( get_english_aggregate_phrase, get_english_select_from_phrase, 
    get_english_order_by_phrase, get_sql_table_fields, get_sql_table_name)


class TestFragments(unittest.TestCase):

    def test_table_name(self):
        
        phrase = get_sql_table_name()
        
        print( "Sample table name:", phrase )
 
    def test_field_names_and_type(self):
        
        selected_fields = get_sql_table_fields(3)
        
        print( "Sample field name:", selected_fields )

    def test_get_english_order_by_phrase(self):
        
        phrase = get_english_order_by_phrase(True)
        
        print( "Sample order by:", phrase )

    def test_get_english_select_from_phrase(self):
        
        phrase = get_english_select_from_phrase()
        
        print( "Sample select from:", phrase )        

    def test_get_english_aggregate_phrase(self):
        
        phrase = get_english_aggregate_phrase()
        
        print( "Sample aggregate:", phrase )     
