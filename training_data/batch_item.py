from dataclasses import dataclass
from typing import List

@dataclass
class BatchItem:
    command_set: int
    table_name: str
    table_fields: List[str]
    create_statement: str
    selected_fields: List[str]
    order_by_fields: List[str] | None
    english_prompt: str
    sql_statement: str

    def print(self):
        print( "Command Set:", self.command_set )
        print( "Table:", self.table_name )
        print( "Table fields:", self.table_fields )
        print( "Create:", self.create_statement )
        print( "Selected fields:", self.selected_fields )
        print( "Order by fields:", self.order_by_fields )
        print( "English:", self.english_prompt )
        print( "SQL:", self.sql_statement )
    