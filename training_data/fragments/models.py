from dataclasses import dataclass
from typing import List


@dataclass
class TableField:
    name: str
    type: str # INTEGER, BIGINT, DECIMAL, NUMERIC, FLOAT, DOUBLE, VARCHAR, CHAR, TEXT, DATE, DATETIME, TIMESTAMP, BOOLEAN, UUID, BLOB, JSON, JSONB


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


@dataclass
class BatchItem:
    command_set: int
    table_name: str
    table_fields: List[TableField]
    create_statement: str
    select: List[SelectField]
    order_by: List[OrderField] | None
    english_prompt: str
    sql_statement: str

    def print(self):
        print( "Command Set:", self.command_set )
        print( "Table name:", self.table_name )
        print( "Table fields:", self.table_fields )
        print( "Create:", self.create_statement )
        print( "Select:", self.select )
        if self.order_by:
            print( "Order by:", self.order_by )
        print( "English:", self.english_prompt )
        print( "SQL:", self.sql_statement )
    