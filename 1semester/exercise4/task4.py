#!/usr/bin/env python3

'''Multiply a 5x3 matrix by a 3x2 matrix (real
matrix product)'''

import numpy as np

arr_5x3 = np.random.rand(5, 3)
arr_3x2 = np.random.rand(3, 2)
prod = np.matmul(arr_5x3, arr_3x2)

print('Matrix 5x3: ', arr_5x3)
print('Matrix 3x2: ', arr_3x2)
print('Product: ', prod)
