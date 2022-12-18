#!/usr/bin/env python3

'''Create a 5x5 array with random values.
and find the minimum and maximum
values'''

import numpy as np

arr = np.random.rand(5, 5)
arr_min = np.amin(arr)
arr_max = np.amax(arr)

print('Array: ', arr, '\n', 'Minimum: ', arr_min, '\n', 'Maximum: ', arr_max)
