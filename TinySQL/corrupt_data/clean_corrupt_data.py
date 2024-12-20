from dataclasses import dataclass
from typing import List, Optional
import random
from TinySQL.training_data.fragments.models import BatchItem, TableField, SelectField

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
    clean_token_str: str = "" # Clean word
    corrupt_token_str: str = "" # Corrupted word
    clean_tokenizer_index: int = UNKNOWN_VALUE # Tokenizer vocab index for clean word
    corrupt_tokenizer_index: int = UNKNOWN_VALUE # Tokenizer vocab index for corrupted word    
    prompt_token_index: int = UNKNOWN_VALUE # Token index in (english or create table) prompt of clean/corrupt word
    answer_token_index: int = UNKNOWN_VALUE # Token index in prediction (sql command) answer of clean/corrupt word
    corrupt_english_prompt: Optional[str] = None
    corrupt_create_statement: Optional[str] = None
    corrupt_sql_statement: Optional[str] = None
    use_corrupt_names: bool = False

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
            create_statement=self.corrupt_create_statement or self.create_statement,
            select=self.select,
            order_by=self.order_by,
            english_prompt=self.corrupt_english_prompt or self.english_prompt,
            sql_statement=self.corrupt_sql_statement or self.sql_statement
        )
    
    def print_clean(self):
        full = self.clean_BatchItem.get_alpaca_prompt() + self.clean_BatchItem.sql_statement 
        print( "Clean: Token=", self.clean_token_str, "TokenizerIndex=", self.clean_tokenizer_index, "PromptTokenIndex=", self.prompt_token_index, "AnswerTokenIndex=", self.answer_token_index, "Prompt+Answer=", full )

    def print_corrupt(self):
        full = self.corrupt_BatchItem.get_alpaca_prompt() + self.corrupt_BatchItem.sql_statement
        print( "Corrupt: Token=", self.corrupt_token_str, "TokenizerIndex=", self.corrupt_tokenizer_index, "PromptTokenIndex=", self.prompt_token_index, "AnswerTokenIndex=", self.answer_token_index, "Prompt+Answer=", full )

    def print_all(self):
        print("Feature name:", self.feature_name)
        if self.feature_name.startswith("Def"):
            print("Clean statement:", self.create_statement)
            print("Corrupt statement:", self.corrupt_create_statement)
        else:
            print("Clean prompt:", self.english_prompt)
            print("Corrupt prompt:", self.corrupt_english_prompt)
        print("Use corrupt names", self.use_corrupt_names)
        print("Clean token:", self.clean_token_str)
        print("Corrupt token:", self.corrupt_token_str)
        print("Prompt token index:", self.prompt_token_index)
        print("Answer token index:", self.answer_token_index)
        print("Clean tokenizer index:", self.clean_tokenizer_index)
        print("Corrupt tokenizer index:", self.corrupt_tokenizer_index)
        print(self.get_alpaca_prompt().replace('\n', ' '))
        print(self.sql_statement.replace('\n', ' '))       

class CorruptFeatureTestGenerator:
    def __init__(self, model_num: int = UNKNOWN_VALUE, cs_num: int = UNKNOWN_VALUE, tokenizer = None, use_corrupt_names: bool = False):
        self.model_num = model_num
        self.cs_num = cs_num
        self.tokenizer = tokenizer
        self.use_corrupt_names = use_corrupt_names

        # Sample data to generate variations
        # For TinyStories all these words translate to 1 token each.
        self.clean_table_names = ["cost", "people", "inventory", "orders", "products"]
        self.corrupt_table_names = ["star", "very", "apple", "blue", "orange"]
        self.clean_field_names = ["price", "count", "amount", "total", "count", "id"]
        self.corrupt_field_names = ["hammer", "little", "wolf", "sky", "yellow"]
        self.clean_field_types = ["INT", "CHAR", "TIME", "TEXT", "JSON"]
    
    def _make_base_item(self) -> BatchItem:
        """Create a random clean base item"""
        table = random.choice(self.clean_table_names)
        fields = random.sample(self.clean_field_names, 2)  # Pick 2 random fields
        types = [random.choice(self.clean_field_types) for _ in fields]
        
        return BatchItem(
            command_set=1,
            table_name=table,
            table_fields=[TableField(f, t) for f, t in zip(fields, types)],
            create_statement=f"CREATE TABLE {table} ( {fields[0]} {types[0]}, {fields[1]} {types[1]} )",
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
            DEFCREATETABLE: self._corrupt_def_create_table,
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

    def tokenize_answer_offset(self):  
        # Llama tokenizes " size" as [128000, 1404] where 128000 is the '<|begin_of_text|>' symbol
        # print(self.tokenizer.convert_ids_to_tokens([128000]))  # Check what `128000` maps to
        # print(self.tokenizer.special_tokens_map)  # Ch
        return 1 if self.model_num == 3 else 0

    def tokenize_text(self, text):
        """Tokenize text and return a token"""       
        answer_offset = self.tokenize_answer_offset()
        token = self.tokenizer(" " + text)["input_ids"][answer_offset] # includes a space
        return token

    def set_clean_corrupt_tokens(self, item: CorruptibleBatchItem, clean_token: str, corrupt_token: str, second_occurrence: bool):
        """Set the clean and corrupt tokens for an item"""
        item.clean_token_str = clean_token
        item.corrupt_token_str = corrupt_token
        item.use_corrupt_names = self.use_corrupt_names

        if self.tokenizer is not None:      
            item.clean_tokenizer_index = self.tokenize_text(item.clean_token_str)
            item.corrupt_tokenizer_index = self.tokenize_text(item.corrupt_token_str)

            # Check the tokens can be tokenized by the tokenizer
            if item.clean_tokenizer_index >= self.tokenizer.vocab_size or item.corrupt_tokenizer_index >= self.tokenizer.vocab_size:
                item.prompt_token_index = UNKNOWN_VALUE
                item.answer_token_index = UNKNOWN_VALUE 
            else:
                clean_prompt_tokens = self.tokenizer(item.clean_BatchItem.get_alpaca_prompt())["input_ids"]
                clean_answer_tokens = self.tokenizer(item.clean_BatchItem.sql_statement)["input_ids"]

                # Tokenize prompt and answer strings to find the index of the clean token in the tokenized strings.
                if second_occurrence:
                    # Find the second instance of the clean token in clean_prompt_tokens
                    occurrences = [i for i, token in enumerate(clean_prompt_tokens) if token == item.clean_tokenizer_index]
                    if len(occurrences) > 1:
                        item.prompt_token_index = occurrences[1]
                    else:
                        print( len(occurrences), item.clean_tokenizer_index, item.clean_token_str, item.get_alpaca_prompt(), clean_prompt_tokens)
                        raise ValueError("Clean token does not appear twice in the prompt tokens.")

                else:
                    item.prompt_token_index = clean_prompt_tokens.index(item.clean_tokenizer_index)

                # Token position in the predicted answer of the token that may be corrupted
                item.answer_token_index = len(clean_prompt_tokens) + clean_answer_tokens.index(item.clean_tokenizer_index)

                
    def _corrupt_eng_table_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        names = self.corrupt_table_names if self.use_corrupt_names else self.clean_table_names
        wrong_table = random.choice([t for t in names if t != base.table_name])
        corrupted = base.english_prompt.replace(base.table_name, wrong_table)

        item = CorruptibleBatchItem( **vars(base), feature_name=ENGTABLENAME, corrupt_english_prompt=corrupted )
        self.set_clean_corrupt_tokens(item, base.table_name, wrong_table, False)
        return item
     
    def _corrupt_eng_field_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        original_field = base.table_fields[0].name
        names = self.corrupt_field_names if self.use_corrupt_names else self.clean_field_names
        wrong_field = random.choice([f for f in names if f != original_field])
        corrupted = base.english_prompt.replace(original_field, wrong_field)

        item = CorruptibleBatchItem( **vars(base), feature_name=ENGFIELDNAME, corrupt_english_prompt=corrupted )
        self.set_clean_corrupt_tokens(item, original_field, wrong_field, False)
        return item    
    
    def _corrupt_def_create_table(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # English prompt may contain "TABLE" so model needs to find "TABLE" after "CREATE" to identify the start of the table name. So we corrupt "CREATE"
        wrong_starts = ["MAKE", "BUILD", "GENERATE", "CONSTRUCT"]
        wrong_start = random.choice(wrong_starts)
        corrupted = base.create_statement.replace("CREATE", wrong_start)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFCREATETABLE, corrupt_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, "CREATE", wrong_start, False)    
        return item

    def _corrupt_def_table_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        names = self.corrupt_table_names if self.use_corrupt_names else self.clean_table_names        
        wrong_table = random.choice([t for t in names if t != base.table_name])
        corrupted = base.create_statement.replace(base.table_name, wrong_table)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFTABLENAME, corrupt_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, base.table_name, wrong_table, True)
        return item
          
    def _corrupt_def_field_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # Pick a field to corrupt and find a different field name
        original_field = base.table_fields[0].name
        names = self.corrupt_field_names if self.use_corrupt_names else self.clean_field_names        
        wrong_field = random.choice([f for f in names if f != original_field])
        # Replace only in create statement
        corrupted = base.create_statement.replace(original_field, wrong_field)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFFIELDNAME, corrupt_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, original_field, wrong_field, True)
        return item
     
    def _corrupt_def_field_separator(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # Replace the comma with various incorrect separators
        wrong_separators = [":"] # , ".", ";", "&"]
        wrong_separator = random.choice(wrong_separators)
        corrupted = base.create_statement.replace(",", wrong_separator)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFFIELDSEPARATOR, corrupt_create_statement=corrupted )   
        self.set_clean_corrupt_tokens(item, ",", wrong_separator, True)         
        return item
 