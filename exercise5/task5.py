#!/usr/bin/env python3

'''Ex-change 2 columns, use function for it. Sort coulumn by
name'''

import pandas as pd

data_frame = pd.read_csv('data.csv')
print('Original data: \n', data_frame)

columns = list(data_frame.columns)
columns[4], columns[3] = columns[3], columns[4]
print("Data with swapped columns 'Name' and 'Sex': \n", )