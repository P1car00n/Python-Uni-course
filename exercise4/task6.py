#!/usr/bin/env python3

'''Extract the integer part of a random array
using 5 different methods'''

import numpy as np

arr = np.random.rand(5, 5) * 10
print('Original array: \n', arr)

# method 1
int_part = np.modf(arr)[1]
print('Integer part method 1: \n', int_part)

# method 2
int_part = np.trunc(arr)
print('Integer part method 2: \n', int_part)

# method 3
with np.printoptions(precision=0):
    print('Integer part method 3: \n', arr)

# method 4
int_part = np.floor(arr)
print('Integer part method 4: \n', int_part)

# method 5
int_part = np.ceil(arr)
print('Integer part method 5: \n', int_part)
