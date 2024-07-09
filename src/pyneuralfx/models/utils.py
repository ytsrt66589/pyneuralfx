import numpy as np


def center_crop(x, length):
    start = (x.shape[-1] - length) // 2
    stop = start + length
    return x[..., start:stop]

def causal_crop(x, length):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[..., start:stop]

