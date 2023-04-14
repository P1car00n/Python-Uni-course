#!/usr/bin/env python3

'''Get names of the DataFrame columns and sum of losted
values DF'''

import pandas as pd

data_frame = pd.read_csv('data.csv')

print('Columns in the DataFrame: ', end='')
for i in data_frame.columns:
    print(i, end='; ')

print()
print('Sum of the numeric values: ')
print(data_frame.sum(numeric_only=True))
