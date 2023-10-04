import math
from typing import Union

def to_tuple(x: Union[int, list, tuple]) -> tuple:
    return (x, x) if isinstance(x, int) else x

def size_conv2d(input_size, kernel_size=3, stride=1, padding=0):
    return math.floor((input_size - kernel_size + 2 * padding) / stride) + 1

def size_conv2dtranspose(input_size, kernel_size=3, stride=1, padding=0):
    return math.floor((input_size - 1) * stride - 2 * padding + (kernel_size -1) + 1)

def size_maxpool2d(input_size, kernel_size=2, stride=2, padding=0):
    return math.floor((input_size + 2 * padding - (kernel_size -1) -1 ) / stride ) + 1

def collect_batch(x):
    return x[0] if x[0].ndim == 4 else x