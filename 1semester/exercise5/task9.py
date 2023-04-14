#!/usr/bin/env python3

'''For any column create histogram'''

import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv('data.csv')

print('Histogram for the "Survived" column')
data_frame.hist(column='Survived')
plt.show()
