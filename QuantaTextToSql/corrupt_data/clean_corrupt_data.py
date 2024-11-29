from dataclasses import dataclass
from typing import Optional
from ..training_data import BatchItem, TableField, SelectField


@dataclass
class CorruptibleBatchItem(BatchItem):
    # Fields describing how this item is corrupted
    corrupted_feature: str = ""
    corrupted_english_prompt: Optional[str] = None
    corrupted_create_statement: Optional[str] = None
    corrupted_sql_statement: Optional[str] = None

    @property
    def clean_version(self) -> BatchItem:
        """Return the clean BatchItem without corruptions"""
        return BatchItem(
            command_set=self.command_set,
            table_name=self.table_name,
            table_fields=self.table_fields,
            create_statement=self.create_statement,
            select=self.select,
            order_by=self.order_by,
            english_prompt=self.english_prompt,
            sql_statement=self.sql_statement
        )
    
    @property
    def corrupt_version(self) -> BatchItem:
        """Return a BatchItem with corruptions applied"""
        return BatchItem(
            command_set=self.command_set,
            table_name=self.table_name,
            table_fields=self.table_fields,
            create_statement=self.corrupted_create_statement or self.create_statement,
            select=self.select,
            order_by=self.order_by,
            english_prompt=self.corrupted_english_prompt or self.english_prompt,
            sql_statement=self.corrupted_sql_statement or self.sql_statement
        )

# Base test case with just the BatchItem fields
BASE_CLEAN = BatchItem(
    command_set=1,
    table_name="cost",
    table_fields=[
        TableField("price", "NUMERIC"),
        TableField("quantity", "INTEGER")
    ],
    create_statement="CREATE TABLE cost (price NUMERIC, quantity INTEGER)",
    select=[SelectField("price", ""), SelectField("quantity", "")],
    order_by=[],
    english_prompt="show me the price and quantity from the cost table",
    sql_statement="SELECT price, quantity FROM cost"
)

# Feature test cases
FEATURE_TESTS = [
    CorruptibleBatchItem(
        **{k: v for k, v in vars(BASE_CLEAN).items()},  # Only copy BatchItem fields
        corrupted_feature="SqlTableStart",
        corrupted_create_statement="MAKE cost (price NUMERIC, quantity INTEGER)"
    ),
    
    CorruptibleBatchItem(
        **{k: v for k, v in vars(BASE_CLEAN).items()},
        corrupted_feature="SqlTableName",
        corrupted_create_statement="CREATE TABLE wrong_table (price NUMERIC, quantity INTEGER)"
    ),
    
    CorruptibleBatchItem(
        **{k: v for k, v in vars(BASE_CLEAN).items()},
        corrupted_feature="EngTableName",
        corrupted_english_prompt="show me the price and quantity from the wrong_table"
    ),
    
    CorruptibleBatchItem(
        **{k: v for k, v in vars(BASE_CLEAN).items()},
        corrupted_feature="EngFieldName",
        corrupted_english_prompt="show me the wrong_field and quantity from the cost table"
    ),
    
    CorruptibleBatchItem(
        **{k: v for k, v in vars(BASE_CLEAN).items()},
        corrupted_feature="SqlFieldSeparator",
        corrupted_create_statement="CREATE TABLE cost (price NUMERIC quantity INTEGER)"
    )
]