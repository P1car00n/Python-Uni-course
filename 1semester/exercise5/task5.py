#!/usr/bin/env python3

'''Ex-change 2 columns, use function for it. Sort coulumn by
name'''

import pandas as pd


def swap_two_columns(columns, *to_swap: int):
    columns[to_swap[0]], columns[to_swap[1]
                                 ] = columns[to_swap[1]], columns[to_swap[0]]


data_frame = pd.read_csv('data.csv')
columns = list(data_frame.columns)
print('Original data: \n', data_frame)
print('Original columns: \n', columns)

swap_two_columns(columns, 3, 4)
print("Data with swapped columns 'Name' and 'Sex': \n",
      data_frame.reindex(columns=columns))
print("Columns with swapped columns 'Name' and 'Sex': \n", columns)
