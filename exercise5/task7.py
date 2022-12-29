#!/usr/bin/env python3

'''Replay ( Apply) missed values in the Column with average
values.'''

import pandas as pd

data_frame = pd.read_csv('data.csv')
print('Original data: \n', data_frame)
print('Mising values before treatment: \n', data_frame.isna().sum())

data_frame.fillna(data_frame.mean(numeric_only=True), inplace=True)
data_frame.interpolate(inplace=True, method='pad')
print('Mmising values after treatment: \n', data_frame.isna().sum())

print('Treated data: \n', data_frame)
