from dataclasses import dataclass
from typing import List, Optional, Iterator
import random
from QuantaTextToSql.training_data.fragments.models import BatchItem, TableField, SelectField, OrderField

ENGTABLENAME = "EngTableName"
ENGFIELDNAME = "EngFieldName"
SQLTABLESTART = "SqlTableStart"
SQLTABLENAME = "SqlTableName"
SQLFIELDSEPARATOR = "SqlFieldSeparator"


@dataclass
class CorruptibleBatchItem(BatchItem):
    corrupted_feature: str = ""
    corrupted_english_prompt: Optional[str] = None
    corrupted_create_statement: Optional[str] = None
    corrupted_sql_statement: Optional[str] = None

    @property
    def clean_version(self) -> BatchItem:
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

class FeatureTestGenerator:
    def __init__(self):
        # Sample data to generate variations
        self.table_names = ["cost", "sales", "inventory", "orders", "products"]
        self.field_names = ["price", "quantity", "amount", "total", "count", "id"]
        self.field_types = ["NUMERIC", "INTEGER", "VARCHAR", "TEXT"]
    
    def _make_base_item(self) -> BatchItem:
        """Create a random clean base item"""
        table = random.choice(self.table_names)
        fields = random.sample(self.field_names, 2)  # Pick 2 random fields
        types = [random.choice(self.field_types) for _ in fields]
        
        return BatchItem(
            command_set=1,
            table_name=table,
            table_fields=[TableField(f, t) for f, t in zip(fields, types)],
            create_statement=f"CREATE TABLE {table} ({fields[0]} {types[0]}, {fields[1]} {types[1]})",
            select=[SelectField(f, "") for f in fields],
            order_by=[],
            english_prompt=f"show me the {fields[0]} and {fields[1]} from the {table} table",
            sql_statement=f"SELECT {fields[0]}, {fields[1]} FROM {table}"
        )

    def generate_feature_examples(self, feature: str, n: int) -> List[CorruptibleBatchItem]:
        """Generate n examples of a specific feature test"""
        generators = {
            ENGTABLENAME: self._corrupt_eng_table_name,
            ENGFIELDNAME: self._corrupt_eng_field_name,
            SQLTABLESTART: self._corrupt_create_table_start,
            SQLTABLENAME: self._corrupt_create_table_name,
            SQLFIELDSEPARATOR: self._corrupt_field_separator,
        }

        if feature not in generators:
            raise ValueError(f"Unknown feature: {feature}")
        
        return [generators[feature]() for _ in range(n)]

    def _corrupt_create_table_start(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        wrong_starts = ["MAKE", "BUILD", "GENERATE", "CONSTRUCT"]
        wrong_start = random.choice(wrong_starts)
        corrupted = base.create_statement.replace("CREATE TABLE", wrong_start)
        return CorruptibleBatchItem(
            **vars(base),
            corrupted_feature=SQLTABLESTART,
            corrupted_create_statement=corrupted
        )

    def _corrupt_create_table_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        wrong_table = random.choice([t for t in self.table_names if t != base.table_name])
        corrupted = base.create_statement.replace(base.table_name, wrong_table)
        return CorruptibleBatchItem(
            **vars(base),
            corrupted_feature=SQLTABLENAME,
            corrupted_create_statement=corrupted
        )

    def _corrupt_eng_table_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        wrong_table = random.choice([t for t in self.table_names if t != base.table_name])
        corrupted = base.english_prompt.replace(base.table_name, wrong_table)
        return CorruptibleBatchItem(
            **vars(base),
            corrupted_feature=ENGTABLENAME,
            corrupted_english_prompt=corrupted
        )

    def _corrupt_eng_field_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        original_field = base.table_fields[0].name
        wrong_field = random.choice([f for f in self.field_names if f != original_field])
        corrupted = base.english_prompt.replace(original_field, wrong_field)
        return CorruptibleBatchItem(
            **vars(base),
            corrupted_feature=ENGFIELDNAME,
            corrupted_english_prompt=corrupted
        )

    def _corrupt_field_separator(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        corrupted = base.create_statement.replace(",", "")
        return CorruptibleBatchItem(
            **vars(base),
            corrupted_feature=SQLFIELDSEPARATOR,
            corrupted_create_statement=corrupted
        )
