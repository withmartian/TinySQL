import unittest

from QuantaTextToSql.training_data import ( get_english_aggregate_phrase, get_english_select_from_phrase, 
    get_english_order_by_phrase, get_sql_table_fields, get_sql_table_name,
    get_english_aggregate_count, get_field_names_count, get_sql_table_name_count, get_english_select_from_count, get_english_order_by_count)


class TestFragments(unittest.TestCase):

    def test_table_name(self):
        
        phrase = get_sql_table_name()
        
        print( "Sample table name:", phrase )
 
    def test_field_names_and_type(self):
        
        selected_fields = get_sql_table_fields("notes", 3)
        
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

    def test_count(self):
        
        t = get_sql_table_name_count()
        f = get_field_names_count()
        s = get_english_select_from_count()
        a = get_english_aggregate_count()
        o = get_english_order_by_count()

        print( "Table name count:", t)
        print( "Field name count:", f)
        print( "English select phrases count:", s)
        print( "English aggregate phrases count:", a)
        print( "English order by phrases count:", o)
        print( "Product:", t*f*s*a*o)
        
