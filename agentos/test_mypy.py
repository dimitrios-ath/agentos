from typing import Literal
from agentos.specs import ParameterSetSpec

class A:
    pass

a: Literal = 0x12b
b: Literal = "hi"
c: Literal = 1
d: Literal = [1, "hi"]
e: Literal = {A: d}
