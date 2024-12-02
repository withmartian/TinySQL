from dataclasses import dataclass
from typing import List, Optional
import random
from QuantaTextToSql.training_data.fragments.models import BatchItem, TableField, SelectField

UNKNOWN_VALUE = -1

ENGTABLENAME = "EngTableName"
ENGFIELDNAME = "EngFieldName"
DEFCREATETABLE = "DefCreateTable"
DEFTABLENAME = "DefTableName"
DEFFIELDNAME = "DefFieldName"
DEFFIELDSEPARATOR = "DefFieldSeparator"

@dataclass
class CorruptibleBatchItem(BatchItem):
    feature_name: str = ""
    clean_token: str = "" # Clean word
    corrupt_token: str = "" # Corrupted word   
    clean_token_index: int = UNKNOWN_VALUE # Tokenizer index for clean word
    corrupt_token_index: int = UNKNOWN_VALUE # Tokenizer index for corrupted word    
    corrupted_english_prompt: Optional[str] = None
    corrupted_create_statement: Optional[str] = None
    corrupted_sql_statement: Optional[str] = None

    @property
    def clean_BatchItem(self) -> BatchItem:
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
    def corrupt_BatchItem(self) -> BatchItem:
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

class CorruptFeatureTestGenerator:
    def __init__(self, model_num: int = UNKNOWN_VALUE, cs_num: int = UNKNOWN_VALUE, tokenizer = None):
        self.model_num = model_num
        self.cs_num = cs_num
        self.tokenizer = tokenizer

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

    def get_generators(self):
        """Return all the generators"""
        generators = {
            ENGTABLENAME: self._corrupt_eng_table_name,
            ENGFIELDNAME: self._corrupt_eng_field_name,
            DEFCREATETABLE: self._corrupt_def_table_start,
            DEFTABLENAME: self._corrupt_def_table_name,
            DEFFIELDSEPARATOR: self._corrupt_def_field_separator,
            DEFFIELDNAME: self._corrupt_def_field_name
        }
        
        return generators

    def generate_feature_examples(self, feature_name: str, batch_size: int = 5) -> List[CorruptibleBatchItem]:
        """Generate n examples of a specific feature test"""
        generators = self.get_generators()
        
        if feature_name not in generators:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        return [generators[feature_name]() for _ in range(batch_size)]

    def safe_tokenize(self, text):
        """Tokenize text and return token"""
        if self.tokenizer is None:
            return UNKNOWN_VALUE
        
        # Llama tokenizes " size" as [128000, 1404] where 128000 is the '<|begin_of_text|>' symbol
        # print(self.tokenizer.convert_ids_to_tokens([128000]))  # Check what `128000` maps to
        # print(self.tokenizer.special_tokens_map)  # Ch
        answer_offset = 1 if self.model_num == 3 else 0

        token = self.tokenizer(" " + text)["input_ids"][answer_offset] # includes a space
        return token

    def set_clean_corrupt_tokens(self, item: CorruptibleBatchItem, clean_token: str, corrupt_token: str):
        """Set the clean and corrupt tokens for an item"""
        item.clean_token = clean_token
        item.corrupt_token = corrupt_token
        item.clean_token_index = self.safe_tokenize(clean_token)
        item.corrupt_token_index = self.safe_tokenize(corrupt_token)

    def _corrupt_def_table_start(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # English prompt may contain "TABLE" so model needs to find "TABLE" after "CREATE" to identify the start of the table name. So we corrupt "CREATE"
        wrong_starts = ["MAKE", "BUILD", "GENERATE", "CONSTRUCT"]
        wrong_start = random.choice(wrong_starts)
        corrupted = base.create_statement.replace("CREATE", wrong_start)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFCREATETABLE, corrupted_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, "CREATE", wrong_start)    
        return item

    def _corrupt_def_table_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        wrong_table = random.choice([t for t in self.table_names if t != base.table_name])
        corrupted = base.create_statement.replace(base.table_name, wrong_table)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFTABLENAME, corrupted_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, base.table_name, wrong_table)
        return item
    
    def _corrupt_eng_table_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        wrong_table = random.choice([t for t in self.table_names if t != base.table_name])
        corrupted = base.english_prompt.replace(base.table_name, wrong_table)

        item = CorruptibleBatchItem( **vars(base), feature_name=ENGTABLENAME, corrupted_english_prompt=corrupted )
        self.set_clean_corrupt_tokens(item, base.table_name, wrong_table)
        return item
     
    def _corrupt_eng_field_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        original_field = base.table_fields[0].name
        wrong_field = random.choice([f for f in self.field_names if f != original_field])
        corrupted = base.english_prompt.replace(original_field, wrong_field)

        item = CorruptibleBatchItem( **vars(base), feature_name=ENGFIELDNAME, corrupted_english_prompt=corrupted )
        self.set_clean_corrupt_tokens(item, original_field, wrong_field)
        return item     
       
    def _corrupt_def_field_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # Pick a field to corrupt and find a different field name
        original_field = base.table_fields[0].name
        wrong_field = random.choice([f for f in self.field_names if f != original_field])
        # Replace only in create statement
        corrupted = base.create_statement.replace(original_field, wrong_field)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFFIELDNAME, corrupted_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, original_field, wrong_field)
        return item
     
    def _corrupt_def_field_separator(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # Replace the comma with various incorrect separators
        wrong_separators = [" ", ";", "|", "&"]
        wrong_separator = random.choice(wrong_separators)
        corrupted = base.create_statement.replace(",", wrong_separator)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFFIELDSEPARATOR, corrupted_create_statement=corrupted )   
        self.set_clean_corrupt_tokens(item, ",", wrong_separator)         
        return item
 