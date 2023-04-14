#!/usr/bin/env python3

'''Consider a random vector with shape
(100,2) representing coordinates, find
point by point distances'''

import numpy as np

vtor = np.random.rand(100, 2)
print('Coordinates: \n', vtor)

dstn = np.array([np.sqrt((vtor[i][0] - vtor[i - 1][0])**2 + \
                (vtor[i][1] - vtor[i - 1][1])**2) for i in range(1, len(vtor))])
print('Distances: \n', dstn)
