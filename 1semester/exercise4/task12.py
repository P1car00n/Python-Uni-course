#!/usr/bin/env python3

'''How to I sort an array by the nth column?'''

import numpy as np

arr = np.random.rand(5, 5)
print('Array: \n', arr)

column = int(input('By which column do you want it to be sorted? '))
sorted_arr = arr[arr[:, column - 1].argsort(axis=0)]

print('Sorted array: \n', sorted_arr)
