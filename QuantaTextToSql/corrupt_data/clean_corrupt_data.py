from dataclasses import dataclass, replace
from typing import List, Callable
from ..training_data import BatchItem, TableField, SelectField

@dataclass
class FeatureTest:
    name: str
    clean_generator: Callable[[], BatchItem]
    corrupt_generator: Callable[[], BatchItem]

# Base test case to minimize duplication
BASE_ITEM = BatchItem(
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

def make_feature_test(name: str, clean_item: BatchItem, **corrupt_changes) -> FeatureTest:
    """Helper to create a feature test with minimal code"""
    return FeatureTest(
        name=name,
        clean_generator=lambda: clean_item,
        corrupt_generator=lambda: replace(clean_item, **corrupt_changes)
    )

# Feature tests
FEATURE_TESTS = [
    make_feature_test(
        "CreateTableStart",
        BASE_ITEM,
        create_statement="MAKE cost (price NUMERIC, quantity INTEGER)"  # Corrupted CREATE
    ),
    
    make_feature_test(
        "CreateTableName",
        BASE_ITEM,
        create_statement="CREATE TABLE wrong_table (price NUMERIC, quantity INTEGER)"
    ),
    
    make_feature_test(
        "EngTableName",
        BASE_ITEM,
        english_prompt="show me the price and quantity from the wrong_table"
    ),
    
    make_feature_test(
        "EngFieldName",
        BASE_ITEM,
        english_prompt="show me the wrong_field and quantity from the cost table"
    ),
    
    make_feature_test(
        "CreateFieldSeparator",
        BASE_ITEM,
        create_statement="CREATE TABLE cost (price NUMERIC quantity INTEGER)"  # Missing comma
    )
]

def generate_test_batch(feature_test: FeatureTest, num_examples: int = 5) -> List[tuple[BatchItem, BatchItem]]:
    """Generate pairs of clean and corrupt examples for a given feature"""
    return [(feature_test.clean_generator(), feature_test.corrupt_generator()) 
            for _ in range(num_examples)]

def get_clean_corrupt_data():
    # Test all features
    for feature in FEATURE_TESTS:
        print(f"\nTesting feature: {feature.name}")
        clean, corrupt = generate_test_batch(feature, num_examples=1)[0]
        
        if feature.name.startswith("Create"):
            print(f"Clean create: {clean.create_statement}")
            print(f"Corrupt create: {corrupt.create_statement}")
        else:
            print(f"Clean prompt: {clean.english_prompt}")
            print(f"Corrupt prompt: {corrupt.english_prompt}")