#!/usr/bin/env python3

'''Normalize a 5x5 random matrix'''

import numpy as np
import sklearn.preprocessing as skp

arr = np.random.rand(5, 5)
arr_n = skp.normalize(arr)

print('Matrix: ', arr)
print('Normalised matrix: ', arr_n)
