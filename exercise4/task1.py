#!/usr/bin/env python3

'''Create a vector with values ranging from
10 to 49. Reverse a vector (first element
becomes last)'''

import numpy as np

vtor = np.array([x for x in range(10, 50)])
vtor_r = np.flip(vtor)

print(vtor_r)
