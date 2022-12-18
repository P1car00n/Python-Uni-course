#!/usr/bin/env python3

'''Create a structured array representing a
position (x,y) and a color (r,g,b)'''

import numpy as np

arr = np.zeros(1, dtype=[('position', (float, 2)), ('color', (float, 3))])

print('Structured array: \n', arr)
print('Color: \n', arr['color'])
print('Position: \n', arr['position'])
