#!/usr/bin/env python3

'''How to get the dates of yesterday, today
and tomorrow?'''

import numpy as np

td = np.datetime64('today')
tm = td + np.timedelta64(1, 'D')
ys = td - np.timedelta64(1, 'D')

print('Today: ', td)
print('Tomorrow: ', tm)
print('Yesterday: ', ys)
