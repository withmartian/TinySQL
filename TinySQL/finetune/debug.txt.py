(sql) dhruv_gretel_ai@a2-ultragpu-4g-dhruv:$ python
>>> from datasets import load_dataset
>>> dataset = load_dataset("withmartian/cs1_dataset")
>>> example = dataset["train"][1]
>>> print(f"Raw SQL Statement: {example['sql_statement']}")
Raw SQL Statement: SELECT
    signature,
    brand,
    blog_id,
    level
FROM permissions
>>> example['sql_statement']
'SELECT\n    signature,\n    brand,\n    blog_id,\n    level\nFROM permissions'

