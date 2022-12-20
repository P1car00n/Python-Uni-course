#!/usr/bin/env python3

'''Subtract the mean of each row of a matrix'''

import numpy as np

matrix = np.random.rand(5, 5)

print('Matrix: \n', matrix)
print('Matrix minus the mean of each row: \n', matrix - matrix.mean(axis=1))
