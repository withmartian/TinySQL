from dataclasses import dataclass
from typing import List, Optional
import random
from TinySQL.training_data.fragments.models import TableName, BatchItem, TableField, SelectField
from TinySQL.training_data.sql_create_table import get_sql_create_table_from_selected_fields
from TinySQL.training_data.sql_create_table import get_sql_create_table
from TinySQL.training_data.sql_select_from import get_sql_select_from
from TinySQL.training_data.sql_order_by import get_sql_order_by
from TinySQL.training_data.fragments.field_names import get_FieldInfo
from TinySQL.training_data.generate_cs1 import evaluate_cs1_prediction_score, get_english_select_from, get_english_order_by, trim_newlines_and_multiple_spaces, evaluate_unrecognised_words
from TinySQL.training_data.fragments.english_order_by import get_english_order_by_phrase
from TinySQL.training_data.fragments.english_aggregates import get_english_sum_phrases, get_english_avg_phrases, get_english_min_phrases, get_english_max_phrases, get_english_count_phrases


field_info = get_FieldInfo()

UNKNOWN_VALUE = -1
ENGTABLENAME = "EngTableName"
ENGFIELDNAME = "EngFieldName"
DEFCREATETABLE = "DefCreateTable"
DEFTABLENAME = "DefTableName"
DEFFIELDNAME = "DefFieldName"
DEFFIELDSEPARATOR = "DefFieldSeparator"
DEFORDERBYFIELD1 = "DefOrderByField1"
DEFORDERBYFIELD2 = "DefOrderByField2"
DEFORDERBYDIRECTION1 = "DefOrderByDirection1"
DEFORDERBYDIRECTION2 = "DefOrderByDirection2" 
DEFAGGREGATEFUNCTION1 = "DefAggregateFunction1"
DEFAGGREGATEFUNCTION2 = "DefAggregateFunction2"
DEFAGGREGATEFIELD1 = "DefAggregateField1"
DEFAGGREGATEFIELD2 = "DefAggregateField2"

DEFWHEREFIELD1 = "DefWhereField1"
DEFJOINFIELD1 = "DefJoinField1"

# CorruptFeatureTestGenerator generates clean and corrupt data for testing.
# That clean and corrupt examples have the same number of tokens under a range of different conditions.   
@dataclass
class CorruptibleBatchItem(BatchItem):
    feature_name: str = ""
    clean_token_str: str = "" # Clean word
    corrupt_token_str: str = "" # Corrupted word
    clean_tokenizer_index: int = UNKNOWN_VALUE # Tokenizer vocab index for clean word
    corrupt_tokenizer_index: int = UNKNOWN_VALUE # Tokenizer vocab index for corrupted word    
    prompt_token_index: int = UNKNOWN_VALUE # Token index in (english or create table) prompt of clean/corrupt word
    answer_token_index: int = UNKNOWN_VALUE # Token index in prediction (sql command) answer of clean/corrupt word
    corrupt_english_prompt: Optional[str] = None # Aka Instruction
    corrupt_create_statement: Optional[str] = None # Aka Context
    corrupt_sql_statement: Optional[str] = None # Aka Response or answer
    use_novel_names: bool = False # Use words not seen in training for the corrupt token

    def corrupt_data_sanity_check(self):
        if self.feature_name.startswith("Def"):
            assert self.corrupt_create_statement is not None 
            assert self.corrupt_create_statement != self.create_statement
        elif self.feature_name.startswith("Eng"):
            assert self.corrupt_english_prompt is not None 
            assert self.corrupt_english_prompt != self.english_prompt

    @property
    def clean_BatchItem(self) -> BatchItem:
        return BatchItem(
            command_set=self.command_set,
            table_name=TableName(name=self.table_name.name, synonym=self.table_name.synonym, use_synonym=self.table_name.use_synonym),
            table_fields=self.table_fields,
            create_statement=self.create_statement,
            select=self.select,
            order_by=self.order_by,
            order_by_phrase=self.order_by_phrase,
            agg_phrases=self.agg_phrases,
            english_prompt=self.english_prompt,
            sql_statement=self.sql_statement
        )
    
    @property
    def corrupt_BatchItem(self) -> BatchItem:
        self.corrupt_data_sanity_check()

        return BatchItem(
            command_set=self.command_set,
            table_name=TableName(name=self.table_name.name, synonym=self.table_name.synonym, use_synonym=self.table_name.use_synonym),
            table_fields=self.table_fields,
            create_statement=self.corrupt_create_statement or self.create_statement,
            select=self.select,
            order_by=self.order_by,
            order_by_phrase=self.order_by_phrase,
            agg_phrases=self.agg_phrases,
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
            print("Clean statement:", self.clean_token_str,  ":", self.create_statement )
            print("Corrupt statement:", self.corrupt_token_str, ":", self.corrupt_create_statement )
        else:
            print("Clean  :", self.clean_token_str, ":", self.english_prompt)
            print("Corrupt:", self.corrupt_token_str, ":", self.corrupt_english_prompt)
        print("Prompt token index:", self.prompt_token_index)
        print("Answer token index:", self.answer_token_index)
        print("Clean tokenizer index:", self.clean_tokenizer_index)
        print("Corrupt tokenizer index:", self.corrupt_tokenizer_index)
        print("Use novel names:", self.use_novel_names)
     

class CorruptFeatureTestGenerator:
    def __init__(self, model_num: int = 1, cs_num: int = 1, tokenizer = None, 
                 use_novel_names: bool = False, use_order_by: bool = False, use_aggregates:bool = False, use_synonyms_field: bool = False, use_synonyms_table: bool = False, 
                 num_fields: int = 2):
        self.model_num = model_num
        self.cs_num = cs_num
        self.tokenizer = tokenizer
        self.use_novel_names = use_novel_names
        self.use_order_by = use_order_by
        self.use_aggregates = use_aggregates
        self.use_synonyms_field = use_synonyms_field
        self.use_synonyms_table = use_synonyms_table
        self.num_fields = num_fields
        
        # Original sample data
        self.clean_table_names = [
            "people", 
            "inventory", 
            "orders", 
            "products", 
            # "flights", not 1 token 
            # "favorites", 
            # "schedule", 
            "items", 
            "users",
            "links", 
            # "messages", 
            # "countries", 
            # "campaigns"
            ]
        
        self.synonym_table_names = {
            "people": "children", 
            "inventory": "stock", 
            "orders": "requests",
            "products": "goods", 
            #"flights": "trips", 
            #"favorites": "picks", 
            #"schedule": "timetable", 
            "items": "objects",
            "users": "customers", 
            "links": "connections", 
            #"messages": "discussions", 
            #"countries": "nations", 
            #"campaigns": "initiatives"
        }
        
        self.novel_table_names = ["star", "very", "apple", "blue", "orange"]
        
        self.clean_field_names = ["price", "count", "amount", "total", "name", "code", 
                                 "number", "label", "type", "category", "status", 
                                 "title", "date", "value", 
                                 # "quantity", Not 1 token 
                                 "rating", 
                                 "color", "size", "weight", "duration"]
        
        self.synonym_field_names = {
            "price": "cost",
            "count": "quantity",
            "amount": "total",
            "total": "sum",
            "name": "title",
            "code": "reference",
            "number": "identifier",
            "label": "tag",
            "type": "category",
            "category": "class",
            "status": "state",
            "title": "heading",
            "date": "time",
            "value": "amount",
            #"quantity": "volume",
            "rating": "score",
            "color": "shade",
            "size": "dimension",
            "weight": "mass",
            "duration": "period"
        }
        self.novel_field_names = ["hammer", "little", "wolf", "sky", "yellow"]
        self.clean_field_types = ["INT", "CHAR", "TIME", "TEXT", "JSON"]
        
        self.directions = ["ASC", "DESC"]
    
    def _make_base_item(self) -> BatchItem:
        """Create a random clean base item with optional ORDER BY support"""
        clean_str = random.choice(self.clean_table_names)
        clean_syn = self.synonym_table_names[clean_str] if self.use_synonyms_table else clean_str
        table_name = TableName(name=clean_str, synonym=clean_syn, use_synonym=self.use_synonyms_table)

        fields = random.sample(self.clean_field_names, self.num_fields)
        eng_fields = ', '.join([self.synonym_field_names[f] for f in fields[:-1]] if self.use_synonyms_field else fields[:-1]) + ' and ' + (self.synonym_field_names[fields[-1]] if self.use_synonyms_field else fields[-1])
        crt_fields = ', '.join([f for f in fields])
        types = [random.choice(self.clean_field_types) for _ in fields]
        
        selected_fields = [TableField(f, t, self.synonym_field_names[f], use_synonym=self.use_synonyms_field) for f, t in zip(fields, types)]
        
        order_by_clause = ""
        order_by_english = ""
        order_by_fields = []
        order_by_phrase = ""
        
        if self.use_order_by:
            order_by_field = random.choice(fields)
            direction = random.choice(self.directions)
            order_by_clause = f" ORDER BY {order_by_field} {direction}"
            order_by_english = f" ordered by {self.synonym_field_names[order_by_field] if self.use_synonyms_field else order_by_field} in {'descending' if direction == 'DESC' else 'ascending'} order"
            order_by_fields = [SelectField(order_by_field, direction, order_by_field)]
            order_by_phrase = {'descending' if direction == 'DESC' else 'ascending'}

        english_prompt = f"show me the {eng_fields} from the {table_name.synonym if self.use_synonyms_table else table_name.name} table{order_by_english}"
        sql_statement = f"SELECT {crt_fields} FROM {table_name.name}{order_by_clause}"

        batch_item_to_return = BatchItem(
            command_set=2 if self.use_order_by else 1,
            table_name=TableName(name=table_name.name, synonym=table_name.synonym, use_synonym=table_name.use_synonym),
            table_fields=selected_fields,
            create_statement=get_sql_create_table_from_selected_fields(table_name, selected_fields)[2],
            select=[SelectField(f, "", self.synonym_field_names[f]) for f in fields],
            order_by=order_by_fields,
            order_by_phrase = order_by_phrase,
            agg_phrases=[],
            english_prompt=english_prompt,
            sql_statement=sql_statement
        )

        # if self.use_aggregates:
        #     print('TFF', type(selected_fields[0]))
        #     (selected_fields, sql_select_statement) = get_sql_select_from(table_name, selected_fields, self.use_aggregates)
        #     print('TFF', type(selected_fields[0]))
        #     (english_select_from_prompt, table_name, selected_fields) = get_english_select_fro(table_name, selected_fields, self.use_synonyms_table, self.use_synonyms_field)
        #     print('TFFFFFF', type(selected_fields[0]))
        #     if self.use_order_by:
        #         (order_by_fields, sql_order_by_statement) = get_sql_order_by(selected_fields)
        #         english_order_by_prompt = get_english_order_by(order_by_fields)
        #     else:
        #         order_by_fields = []
        #         english_order_by_prompt = ""
        #         sql_order_by_statement = ""

        #     batch_item_to_return = BatchItem(
        #     command_set=2,
        #     table_name=TableName(name=table_name.name, synonym=table_name.synonym, use_synonym=table_name.use_synonym), 
        #     table_fields=selected_fields,
        #     create_statement=get_sql_create_table_from_selected_fields(table_name, selected_fields)[2],
        #     select=selected_fields,
        #     order_by=order_by_fields,
        #     english_prompt=english_select_from_prompt + " " + english_order_by_prompt,
        #     sql_statement=sql_select_statement + " " + sql_order_by_statement, # ground truth
        # )
        

        if self.use_aggregates:
            # Keep the original table_fields for table definition
            original_table_fields = selected_fields
            
            # Get SelectFields for the SELECT statement
            (selected_fields, sql_select_statement) = get_sql_select_from(table_name, original_table_fields, self.use_aggregates)
            (english_select_from_prompt, table_name, selected_fields), agg_phrases = get_english_select_from(table_name, selected_fields, self.use_synonyms_table, self.use_synonyms_field)
            if self.use_order_by:
                (order_by_fields, sql_order_by_statement) = get_sql_order_by(selected_fields)
                english_order_by_prompt, order_by_phrase = get_english_order_by(order_by_fields)
            else:
                order_by_fields = []
                english_order_by_prompt = ""
                sql_order_by_statement = ""  

            batch_item_to_return = BatchItem(
                command_set=2,
                table_name=TableName(name=table_name.name, synonym=table_name.synonym, use_synonym=table_name.use_synonym), 
                table_fields=original_table_fields,  # Use TableFields here
                create_statement=get_sql_create_table_from_selected_fields(table_name, original_table_fields)[2],  # And here
                select=selected_fields,  # Use SelectFields here
                order_by=order_by_fields,
                order_by_phrase = order_by_phrase,
                agg_phrases = agg_phrases,
                english_prompt=english_select_from_prompt + " " + english_order_by_prompt,
                sql_statement=sql_select_statement + " " + sql_order_by_statement,
            )

        
        return batch_item_to_return
    def get_generators(self):
        """Return all the generators"""
        generators = {
            ENGTABLENAME: self._corrupt_eng_table_name,
            ENGFIELDNAME: self._corrupt_eng_field_name,
            DEFCREATETABLE: self._corrupt_def_create_table,
            DEFTABLENAME: self._corrupt_def_table_name,
            DEFFIELDSEPARATOR: self._corrupt_def_field_separator,
            DEFFIELDNAME: self._corrupt_def_field_name,
            DEFORDERBYFIELD1: self._corrupt_def_order_by_field1,
            DEFORDERBYFIELD2: self._corrupt_def_order_by_field2,
            DEFORDERBYDIRECTION1: self._corrupt_def_order_by_direction1,
            DEFORDERBYDIRECTION2: self._corrupt_def_order_by_direction2,
            DEFAGGREGATEFUNCTION1: self._corrupt_def_aggregate_function1,
            DEFAGGREGATEFUNCTION2: self._corrupt_def_aggregate_function2,
            DEFAGGREGATEFIELD1: self._corrupt_def_aggregate_field1,
            DEFAGGREGATEFIELD2: self._corrupt_def_aggregate_field2
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

    def set_clean_corrupt_tokens(self, item: CorruptibleBatchItem, clean_token: str, corrupt_token: str, answer_token: str, second_occurrence: bool):
        """Set the clean and corrupt tokens for an item"""
        item.clean_token_str = clean_token
        item.corrupt_token_str = corrupt_token
        item.use_novel_names = self.use_novel_names
        

        if self.tokenizer is not None:      
            item.clean_tokenizer_index = self.tokenize_text(item.clean_token_str)
            item.corrupt_tokenizer_index = self.tokenize_text(item.corrupt_token_str)

            # Check the tokens can be tokenized by the tokenizer
            if item.clean_tokenizer_index >= self.tokenizer.vocab_size or item.corrupt_tokenizer_index >= self.tokenizer.vocab_size:
                item.prompt_token_index = UNKNOWN_VALUE
                item.answer_token_index = UNKNOWN_VALUE 
            else:
                clean_prompt_str = item.clean_BatchItem.get_alpaca_prompt()
                clean_answer_str = item.clean_BatchItem.sql_statement           
                clean_prompt_tokens = self.tokenizer(clean_prompt_str)["input_ids"]
                clean_answer_tokens = self.tokenizer(clean_answer_str)["input_ids"]                

                # Tokenize prompt and answer strings to find the index of the clean token in the tokenized strings.
                if second_occurrence and not self.use_synonyms_field and not self.use_synonyms_table:
                    # Find the second instance of the clean token in clean_prompt_tokens
                    occurrences = [i for i, token in enumerate(clean_prompt_tokens) if token == item.clean_tokenizer_index]
                    if len(occurrences) > 1:
                        item.prompt_token_index = occurrences[1]
                    elif (item.feature_name == DEFFIELDSEPARATOR) and len(occurrences) == 1:
                        # May have "Instruction: pull a and B from c Context: Create table c { a INT, b INT }" so comma is first occurrence.
                        item.prompt_token_index = occurrences[0]
                    else:
                        print( len(occurrences), item.clean_tokenizer_index, item.clean_token_str, item.get_alpaca_prompt(), clean_prompt_tokens)
                        raise ValueError("Clean token does not appear twice in the prompt tokens.")

                else:
                    item.prompt_token_index = clean_prompt_tokens.index(item.clean_tokenizer_index)

                # If we are using a semantic model, and testing feature EngTableName or EngFieldName, then the clean token will not appear in the answer 
                answer_tokenizer_index = self.tokenize_text(answer_token)
                item.answer_token_index = len(clean_prompt_tokens) + clean_answer_tokens.index(answer_tokenizer_index) - 1

                
    def _corrupt_eng_table_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        table_str = base.table_name.name_str
        names = self.novel_table_names if self.use_novel_names else self.clean_table_names
        wrong_table = random.choice([t for t in names if t != table_str])
        corrupted = base.english_prompt.replace(table_str, wrong_table)
        item = CorruptibleBatchItem( **vars(base), feature_name=ENGTABLENAME, corrupt_english_prompt=corrupted )
        # For semantic models, the clean token will not appear in the answer
        self.set_clean_corrupt_tokens(item, table_str, wrong_table, base.table_name.name, False)
        return item
     
    def _corrupt_eng_field_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        field_str = base.table_fields[0].name_str
        base_fields = [field.synonym for field in base.table_fields] if self.use_synonyms_field else [field.name for field in base.table_fields]  
        names = self.novel_field_names if self.use_novel_names else self.clean_field_names
        wrong_field = random.choice([f for f in names if f not in base_fields])
        corrupted = base.english_prompt.replace(field_str, wrong_field)

        item = CorruptibleBatchItem( **vars(base), feature_name=ENGFIELDNAME, corrupt_english_prompt=corrupted )
        # For semantic models, the clean token will not appear in the answer
        self.set_clean_corrupt_tokens(item, field_str, wrong_field, base.table_fields[0].name, False)
        return item    
    
    def _corrupt_def_create_table(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # English prompt may contain "TABLE" so model needs to find "TABLE" after "CREATE" to identify the start of the table name. So we corrupt "CREATE"
        wrong_starts = ["MAKE", "BUILD", "GENERATE", "CONSTRUCT"]
        wrong_start = random.choice(wrong_starts)
        corrupted = base.create_statement.replace("CREATE", wrong_start)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFCREATETABLE, corrupt_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, "CREATE", wrong_start, wrong_start, False)    
        return item

    def _corrupt_def_table_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        names = self.novel_table_names if self.use_novel_names else self.clean_table_names        
        wrong_table = random.choice([t for t in names if t != base.table_name.name])
        corrupted = base.create_statement.replace(base.table_name.name, wrong_table)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFTABLENAME, corrupt_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, base.table_name.name, wrong_table, base.table_name.name, True)
        return item
          
    def _corrupt_def_field_name(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # Pick a field to corrupt and find a different field name
        original_field = base.table_fields[0].name
        names = self.novel_field_names if self.use_novel_names else self.clean_field_names      
        base_fields = [field.name for field in base.table_fields]  
        wrong_field = random.choice([f for f in names if f not in base_fields])
        # Replace only in create statement
        corrupted = base.create_statement.replace(original_field, wrong_field)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFFIELDNAME, corrupt_create_statement=corrupted )
        self.set_clean_corrupt_tokens(item, original_field, wrong_field, original_field, True)
        return item
     
    def _corrupt_def_field_separator(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        # Replace the comma with various incorrect separators
        wrong_separators = [":"] # , ".", ";", "&"]
        wrong_separator = random.choice(wrong_separators)
        corrupted = base.create_statement.replace(",", wrong_separator)

        item = CorruptibleBatchItem( **vars(base), feature_name=DEFFIELDSEPARATOR, corrupt_create_statement=corrupted )   
        self.set_clean_corrupt_tokens(item, ",", wrong_separator, ",", True)         
        return item

    def _corrupt_def_order_by_field1(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        if not self.use_order_by or not base.order_by:
            # Try again if we don't have an ORDER BY
            return self._corrupt_def_order_by_field()
            
        original_field = base.order_by[0].name
        print(base.order_by)
        # Get possible fields excluding the current order by field
        names = self.novel_field_names if self.use_novel_names else self.clean_field_names
        #base_fields = [field.name for field in base.table_fields]
        wrong_field = random.choice([f for f in names if f != original_field])
        
        # Corrupt the ORDER BY field in SQL statement
        corrupted = base.sql_statement.replace(f"ORDER BY {original_field}", f"ORDER BY {wrong_field}")
        parts = base.english_prompt.rsplit(original_field, 1)
        if len(parts) > 1:
            corrupted_eng = wrong_field.join(parts)
        else:
            corrupted_eng = base.english_prompt

        corrupted_eng = base.english_prompt.replace(original_field, wrong_field)
        item = CorruptibleBatchItem(**vars(base), 
                                feature_name=DEFORDERBYFIELD1,
                                corrupt_english_prompt=corrupted_eng,
                                corrupt_sql_statement=corrupted)
        self.set_clean_corrupt_tokens(item, original_field, wrong_field, original_field, False)
        return item
    

    def _corrupt_def_order_by_field2(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        if not self.use_order_by or not base.order_by:
            # Try again if we don't have an ORDER BY
            return self._corrupt_def_order_by_field()
        if len(base.order_by) < 2:
            original_field = base.order_by[0].name
        else:
            original_field = base.order_by[1].name
        # Get possible fields excluding the current order by field
        names = self.novel_field_names if self.use_novel_names else self.clean_field_names
        #base_fields = [field.name for field in base.table_fields]
        wrong_field = random.choice([f for f in names if f != original_field])
        # Corrupt the ORDER BY field in SQL statement
        corrupted = base.sql_statement.replace(f"ORDER BY {original_field}", f"ORDER BY {wrong_field}")
        parts = base.english_prompt.rsplit(original_field, 1)
        if len(parts) > 1:
            corrupted_eng = wrong_field.join(parts)
        else:
            corrupted_eng = base.english_prompt

        corrupted_eng = base.english_prompt.replace(original_field, wrong_field)
        item = CorruptibleBatchItem(**vars(base), 
                                feature_name=DEFORDERBYFIELD1,
                                corrupt_english_prompt=corrupted_eng,
                                corrupt_sql_statement=corrupted)
        self.set_clean_corrupt_tokens(item, original_field, wrong_field, original_field, False)
        return item
    

    def _corrupt_def_order_by_direction1(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        if not self.use_order_by or not base.order_by:
            # Try again if we don't have an ORDER BY
            return self._corrupt_def_order_by_direction()
        
        original_direction = (base.order_by[0].aggregate == "ASC") if isinstance(base.order_by[0], SelectField) else base.order_by[0].asc# Direction is stored in aggregate field
        # Get opposite direction
        wrong_direction = "DESC" if original_direction else "ASC"
        orig_direct = "ASC" if original_direction else "DESC"
        # Corrupt the direction in SQL statement
        parts = base.sql_statement.split(f"{orig_direct}", 1)
        corrupted = wrong_direction.join(parts)
        original_eng_order = base.order_by_phrase[0]
        wrong_order = get_english_order_by_phrase(not original_direction)
        corrupted_eng = base.english_prompt.replace(original_eng_order, wrong_order)
        item = CorruptibleBatchItem(**vars(base),
                                feature_name=DEFORDERBYDIRECTION1,
                                corrupt_sql_statement=corrupted,
                                corrupt_english_prompt=corrupted_eng)
        self.set_clean_corrupt_tokens(item, original_direction, wrong_direction, original_direction, False)
        return item


    def _corrupt_def_order_by_direction2(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        if not self.use_order_by or not base.order_by:
            # Try again if we don't have an ORDER BY
            return self._corrupt_def_order_by_direction()
        if len(base.order_by) < 2:
            original_direction = (base.order_by[0].aggregate == "ASC") if isinstance(base.order_by[0], SelectField) else base.order_by[0].asc
        else:
            original_direction = (base.order_by[1].aggregate == "ASC") if isinstance(base.order_by[1], SelectField) else base.order_by[1].asc# Direction is stored in aggregate field
        # Get opposite direction
        wrong_direction = "DESC" if original_direction else "ASC"
        orig_direct = "ASC" if original_direction else "DESC"
        # Corrupt the direction in SQL statement
        parts = base.sql_statement.rsplit(f"{orig_direct}", 1)
        corrupted = wrong_direction.join(parts)
        if len(base.order_by_phrase) < 2:
            original_eng_order = base.order_by_phrase[0]
        else:
            original_eng_order = base.order_by_phrase[1]
        wrong_order = get_english_order_by_phrase(not original_direction)
        corrupted_eng = base.english_prompt.replace(original_eng_order, wrong_order)
        item = CorruptibleBatchItem(**vars(base),
                                feature_name=DEFORDERBYDIRECTION2,
                                corrupt_sql_statement=corrupted,
                                corrupt_english_prompt=corrupted_eng)
        self.set_clean_corrupt_tokens(item, original_direction, wrong_direction, original_direction, False)
        return item
    

    def _corrupt_def_aggregate_function1(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        if not self.use_aggregates or ('(' not in base.sql_statement):
            print('Corruption impossible')
        
        if '(' in base.sql_statement:
            original_agg = base.sql_statement.split('(')[0].split(' ')[-1]
            field_name = base.sql_statement.split('(')[1].split(')')[0]
            # Get wrong aggregate function
            wrong_agg = random.choice([a for a in ["SUM", "COUNT", "AVG", "MAX", "MIN"] 
                                    if a != original_agg])
            
            # Corrupt the aggregate function in SQL statement
            corrupted = base.sql_statement.replace(f"{original_agg}({field_name})",
                                                f"{wrong_agg}({field_name})")
            
            eng_agg = base.agg_phrases[0]
            if original_agg == "SUM":
                phrases = get_english_sum_phrases()
            elif original_agg == "AVG":
                phrases = get_english_avg_phrases()
            elif original_agg == "MIN":
                phrases = get_english_min_phrases()
            elif original_agg == "MAX":
                phrases = get_english_max_phrases()
            elif original_agg == "COUNT":
                phrases = get_english_count_phrases()
            
            wrong_agg_phrase = random.choice([p for p in phrases if p != eng_agg])
            corrupted_eng = base.english_prompt.replace(eng_agg, wrong_agg_phrase)
    
            item = CorruptibleBatchItem(**vars(base),
                                    feature_name=DEFAGGREGATEFUNCTION1,
                                    corrupt_english_prompt=corrupted_eng,
                                    corrupt_sql_statement=corrupted)
            self.set_clean_corrupt_tokens(item, original_agg, wrong_agg, original_agg, False)
            return item
                
        return self._corrupt_def_aggregate_function1()
    
    def _corrupt_def_aggregate_function2(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        if not self.use_aggregates or ('(' not in base.sql_statement):
            print('Corruption impossible')
        
        if '(' in base.sql_statement:
            parts = base.sql_statement.split('(')
            if len(parts) > 2:  # Check if there's a second parenthesis
                # For first aggregate
                original_agg = parts[1].split(' ')[-1]  # Get agg function before second '('
                field_name = parts[2].split(')')[0]  # Get field in second parentheses
                
            else:
                original_agg = parts[0].split(' ')[-1]
                field_name = parts[1].split(')')[0]

            # Get wrong aggregate function
            wrong_agg = random.choice([a for a in ["SUM", "COUNT", "AVG", "MAX", "MIN"] 
                                    if a != original_agg])
            
            # Corrupt the aggregate function in SQL statement
            corrupted = base.sql_statement.replace(f"{original_agg}({field_name})",
                                                f"{wrong_agg}({field_name})")
            
            eng_agg = base.agg_phrases[0]
            if original_agg == "SUM":
                phrases = get_english_sum_phrases()
            elif original_agg == "AVG":
                phrases = get_english_avg_phrases()
            elif original_agg == "MIN":
                phrases = get_english_min_phrases()
            elif original_agg == "MAX":
                phrases = get_english_max_phrases()
            elif original_agg == "COUNT":
                phrases = get_english_count_phrases()
            
            wrong_agg_phrase = random.choice([p for p in phrases if p != eng_agg])
            corrupted_eng = base.english_prompt.replace(eng_agg, wrong_agg_phrase)
    
            item = CorruptibleBatchItem(**vars(base),
                                    feature_name=DEFAGGREGATEFUNCTION2,
                                    corrupt_english_prompt=corrupted_eng,
                                    corrupt_sql_statement=corrupted)
            self.set_clean_corrupt_tokens(item, original_agg, wrong_agg, original_agg, False)
            return item
                
        return self._corrupt_def_aggregate_function2()
    


    def _corrupt_def_aggregate_field1(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        if not self.use_aggregates or ('(' not in base.sql_statement):
            print('Corruption impossible')
        
        # Find first aggregated field
        if '(' in base.sql_statement:
            agg_func = base.sql_statement.split('(')[0]
            original_field = base.sql_statement.split('(')[1].split(')')[0]
            
            # Get wrong field name
            names = self.novel_field_names if self.use_novel_names else self.clean_field_names
            wrong_field = random.choice([f for f in names if f != original_field])
            
            # Corrupt the field name inside the aggregate
            corrupted = base.sql_statement.replace(f"{agg_func}({original_field})",
                                                f"{agg_func}({wrong_field})")
            
            corrupted_eng = base.english_prompt.replace(original_field, wrong_field)
            
            item = CorruptibleBatchItem(**vars(base),
                                    feature_name=DEFAGGREGATEFIELD1,
                                    corrupt_english_prompt=corrupted_eng,
                                    corrupt_sql_statement=corrupted)
            self.set_clean_corrupt_tokens(item, original_field, wrong_field, original_field, False)
            return item
                
        return self._corrupt_def_aggregate_field1()
    

    def _corrupt_def_aggregate_field2(self) -> CorruptibleBatchItem:
        base = self._make_base_item()
        if not self.use_aggregates or ('(' not in base.sql_statement):
            print('Corruption impossible')
        
        # Find first aggregated field
        if '(' in base.sql_statement:
            parts = base.sql_statement.split('(')
            if len(parts) > 2:  # Check if there's a second parenthesis
                # For first aggregate
                agg_func = parts[1].split(' ')[-1]  # Get agg function before second '('
                original_field = parts[2].split(')')[0]  # Get field in second parentheses
                
            else:
                agg_func = parts[0].split(' ')[-1]
                original_field = parts[1].split(')')[0]
            
            # Get wrong field name
            names = self.novel_field_names if self.use_novel_names else self.clean_field_names
            wrong_field = random.choice([f for f in names if f != original_field])
            
            # Corrupt the field name inside the aggregate
            corrupted = base.sql_statement.replace(f"{agg_func}({original_field})",
                                                f"{agg_func}({wrong_field})")
            
            corrupted_eng = base.english_prompt.replace(original_field, wrong_field)
            
            item = CorruptibleBatchItem(**vars(base),
                                    feature_name=DEFAGGREGATEFIELD2,
                                    corrupt_english_prompt=corrupted_eng,
                                    corrupt_sql_statement=corrupted)
            self.set_clean_corrupt_tokens(item, original_field, wrong_field, original_field, False)
            return item
                
        return self._corrupt_def_aggregate_field2()