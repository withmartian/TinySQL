from .generate_cs1 import generate_cs1
from .generate_cs2 import generate_cs2
from .generate_cs3 import generate_cs3
from .generate_cs4 import generate_cs4
from .generate_cs5 import generate_cs5


def generate_csn(batch_size:int, csn:int,
                 order_by_clause_probability:float=0.9, where_clause_probability:float=0.9,
                 use_aggregates:bool=False, min_cols:int=2, max_cols:int=12,
                 use_synonyms_table:bool=False, use_synonyms_field:bool=False
                 ):
    if csn == 1:
        return generate_cs1(batch_size=batch_size, min_cols=min_cols, max_cols=max_cols, use_synonyms_table=use_synonyms_table, use_synonyms_field=use_synonyms_field)
    
    if csn == 2:
        return generate_cs2(batch_size=batch_size, order_by_clause_probability=order_by_clause_probability, use_aggregates=use_aggregates, min_cols=min_cols, max_cols=max_cols, use_synonyms_table=use_synonyms_table, use_synonyms_field=use_synonyms_field)
    
    if csn == 3:
        return generate_cs3(batch_size=batch_size, order_by_clause_probability=order_by_clause_probability, min_cols=min_cols, max_cols=max_cols, use_synonyms_table=use_synonyms_table, use_synonyms_field=use_synonyms_field)

    if csn == 4:
        return generate_cs4(
            batch_size=batch_size,
            order_by_clause_probability=order_by_clause_probability, where_clause_probability=where_clause_probability,
            min_cols=min_cols, max_cols=max_cols,
            use_synonyms_table=use_synonyms_table, use_synonyms_field=use_synonyms_field
        )

    if csn == 5:
        return generate_cs5(
            batch_size=batch_size,
            order_by_clause_probability=order_by_clause_probability, where_clause_probability=where_clause_probability,
            min_cols=min_cols, max_cols=max_cols,
            use_synonyms_table=use_synonyms_table, use_synonyms_field=use_synonyms_field
        )


    raise ValueError(f"Invalid csn: {csn}")
