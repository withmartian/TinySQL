from dataclasses import dataclass
from typing import List


@dataclass
class TableField:
    """Represents a field and its metadata"""
    name: str
    type: str # INTEGER, BIGINT, DECIMAL, NUMERIC, FLOAT, DOUBLE, VARCHAR, CHAR, TEXT, DATE, DATETIME, TIMESTAMP, BOOLEAN, UUID, BLOB, JSON, JSONB
    synonym: str = None # english synonym for the field name
    use_synonym: bool = False # whether to use the synonym in the Instructions (english statement)

    @property
    def name_str(self) -> str:
        """Return the synonym if `use_synonym` is True, otherwise return the name."""
        return self.synonym if self.use_synonym else self.name
    
@dataclass
class TableName:
    """Represents a table and a synonym"""
    name: str
    synonym: str = None # english synonym for the table name
    use_synonym: bool = False # whether to use the synonym in the Instructions (english statement)

    @property
    def name_str(self) -> str:
        """Return the synonym if `use_synonym` is True, otherwise return the name."""
        return self.synonym if self.use_synonym else self.name

@dataclass
class SelectField:
    name: str
    aggregate: str # SUM, AVG, MIN, MAX, COUNT, ""
    synonym: str = None # english synonym for the field name
    use_synonym: bool = False # whether to use the synonym in the Instructions (english statement)

    @property
    def name_str(self) -> str:
        """Return the synonym if `use_synonym` is True, otherwise return the name."""
        return self.synonym if self.use_synonym else self.name
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
    create_statement: str # aka Context
    select: List[SelectField]
    order_by: List[OrderField]
    order_by_phrase: List[str]
    agg_phrases: List[str]
    english_prompt: str # aka Instruction
    sql_statement: str # aka Response
    where_fields: List[str] = None
    where_literals: List[str] = None
    join_table: str = ""
    join_fields: List[str] = None
    join_condition: List[str] = None
    where: List[str] = None

    def print(self):
        print("Command Set:", self.command_set)
        print("Table name:", self.table_name.name, "with synonym:", self.table_name.synonym)
        print("Table fields:", self.table_fields)
        print("Create:", self.create_statement)
        print("Select:", self.select)
        if self.order_by:
            print("Order by:", self.order_by)
        print("English:", self.english_prompt)
        print("SQL:", self.sql_statement)

        if self.join_table:
            print("Join table:", self.join_table)

        if self.join_conditions:
            print("Join fields:", self.join_conditions)

        if self.where:
            print("Where conditions: ", self.where, "\nWhere fields: ", self.where_fields, "\nWhere literals: ", self.where_literals)

    def get_alpaca_prompt(self):
        alpaca_prompt = """### Instruction: {} ### Context: {} ### Response: """

        # Substitute the english_prompt and create_statement into the alpaca_prompt
        alpaca_prompt = alpaca_prompt.format(self.english_prompt, self.create_statement)

        alpaca_prompt = trim_newlines_and_multiple_spaces(alpaca_prompt) + " "

        return alpaca_prompt