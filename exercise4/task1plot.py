#!/usr/bin/env python3

'''Create a vector with values ranging from
10 to 49. Reverse a vector (first element
becomes last)'''

import numpy as np
import matplotlib.pyplot as plt

values = [x for x in range(10, 50)]

vtor = np.array(values)
vtor_r = np.flip(vtor)

print(vtor_r)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.pie(vtor)
ax2.pie(vtor_r)
ax3.bar(x=values, height=vtor)
ax4.bar(x=values, height=vtor_r)

plt.show()
