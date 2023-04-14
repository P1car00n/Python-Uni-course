#!/usr/bin/env python3

'''Consider two random array A and B, check
if they are equal'''

import numpy as np

arr1 = np.random.rand(1, 5) * 10
arr2 = np.random.rand(1, 5) * 10

arr3 = np.array([x for x in range(1, 5)])
arr4 = np.array([x for x in range(1, 5)])

print('Array 1', arr1)
print('Array 2', arr2)
print('Array 3', arr3)
print('Array 4', arr4)
print('Arrays 1 and 2 are equal: ', np.array_equal(arr1, arr2))
print('Arrays 3 and 4 are equal: ', np.array_equal(arr3, arr4))
