#!/usr/bin/env python3

'''Compute a matrix rank'''

import numpy as np

matrix = input('Hit "Enter" to use a random matrix: ')

if matrix == '':
    matrix = np.random.rand(5, 5)

print('Matrix: \n', matrix)

print('Rank: ', np.linalg.matrix_rank(matrix))
