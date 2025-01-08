from dataclasses import dataclass
from typing import List


@dataclass
class TableField:
    """Represents a field and its metadata"""
    name: str
    type: str # INTEGER, BIGINT, DECIMAL, NUMERIC, FLOAT, DOUBLE, VARCHAR, CHAR, TEXT, DATE, DATETIME, TIMESTAMP, BOOLEAN, UUID, BLOB, JSON, JSONB
    synonym: str # english synonyms for the field name

@dataclass
class TableName:
    """Represents a table and a synonym"""
    name: str
    synonym: str # english synonym for the table name


@dataclass
class SelectField:
    name: str
    aggregate: str # SUM, AVG, MIN, MAX, COUNT, ""

    @property
    def aggregate_of_field(self):
        if self.aggregate == "":
            return self.name
        return f"{self.aggregate}({self.name})"
    
    @property
    def aggregated_name(self):
        if self.aggregate == "":
            return self.name
        return f"{self.aggregate}_{self.name}"
    
    @property
    def sql(self):
        if self.aggregate == "":
            return self.name
        return f"{self.aggregate_of_field} AS {self.aggregated_name}"

@dataclass
class OrderField:
    name: str
    asc: bool


# Remove newlines, multiple spaces, leading/trailing spaces 
def trim_newlines_and_multiple_spaces(statement: str) -> str:
    str_list = statement.replace('\n', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').strip().split()
    return ' '.join(str_list)

@dataclass
class BatchItem:
    command_set: int
    table_name: TableName
    table_fields: List[TableField]
    create_statement: str
    select: List[SelectField]
    #order_by: List[OrderField] | None
    order_by: List[OrderField]
    english_prompt: str
    sql_statement: str

    def print(self):
        print( "Command Set:", self.command_set )
        print( "Table name:", self.table_name.name, "with synonym:", self.table_name.synonym )
        print( "Table fields:", self.table_fields )
        print( "Create:", self.create_statement )
        print( "Select:", self.select )
        if self.order_by:
            print( "Order by:", self.order_by )
        print( "English:", self.english_prompt )
        print( "SQL:", self.sql_statement )

    def get_alpaca_prompt(self):
        alpaca_prompt = """### Instruction: {} ### Context: {} ### Response: """

        # Substitute the english_prompt and create_statement into the alpaca_prompt
        alpaca_prompt = alpaca_prompt.format(self.english_prompt, self.create_statement)

        alpaca_prompt = trim_newlines_and_multiple_spaces(alpaca_prompt) + " "

        return alpaca_prompt
