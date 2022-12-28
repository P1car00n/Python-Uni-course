#!/usr/bin/env python3

'''Change the data in the column of DataFrame according to
some condition'''

import pandas as pd

data_frame = pd.read_csv('data.csv')
print('Original data: \n', data_frame[['PassengerId', 'Sex']])

data_frame.loc[data_frame['Sex'] == 'male', 'Sex'] = 'man'
print('Changed data: \n', data_frame[['PassengerId', 'Sex']])
