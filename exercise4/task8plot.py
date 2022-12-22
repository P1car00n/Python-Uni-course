#!/usr/bin/env python3

'''Consider a generator function that
generates 10 integers and use it to build an
array'''

import numpy as np
import matplotlib.pyplot as plt


def generate():
    for i in range(10):
        yield i


arr = np.fromiter(generate(), int)

print(arr)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.pie(arr)
ax2.bar(x=arr, height=arr)

plt.show()
