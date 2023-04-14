#!/usr/bin/env python3

'''Consider a 16x16 array, how to get the
block-sum (block size is 4x4)'''

import numpy as np

arr = np.ones((16, 16))
print('16x16 array: \n', arr)

block_size = 4

arr_split = np.lib.stride_tricks.sliding_window_view(
    arr, (block_size, block_size))[::block_size, ::block_size]
print('Block-sum: \n', arr_split.sum((0, 1)))
