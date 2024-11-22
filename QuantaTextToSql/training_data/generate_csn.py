from .generate_cs1 import generate_cs1
from .generate_cs2 import generate_cs2
from .generate_cs3 import generate_cs3

def generate_csn(batch_size:int, csn:int, order_by_clause_probability:float=0.9, use_aggregates:bool=False):
    if csn == 1:
        return generate_cs1(batch_size)
    elif csn == 2:
        return generate_cs2(batch_size, order_by_clause_probability, use_aggregates)
    elif csn == 3:
        return generate_cs3(batch_size, order_by_clause_probability)
    else:
        raise ValueError(f"Invalid csn: {csn}")
