#!/usr/bin/env python3

'''Consider a generator function that
generates 10 integers and use it to build an
array'''

import numpy as np


def generate():
    for i in range(10):
        yield i


arr = np.fromiter(generate(), int)

print(arr)
