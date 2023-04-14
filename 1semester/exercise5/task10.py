#!/usr/bin/env python3

'''Create Correlation Matrix for any column'''

import pandas as pd

data_frame = pd.read_csv('data.csv')

print('Correlation between the columns "Age" and "Survived": ',
      data_frame['Age'].corr(data_frame['Survived']))
