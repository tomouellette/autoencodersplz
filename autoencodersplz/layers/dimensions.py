from typing import Union
from einops import rearrange, repeat

def to_tuple(x: Union[int, list, tuple]) -> tuple:
    return (x, x) if isinstance(x, int) else x