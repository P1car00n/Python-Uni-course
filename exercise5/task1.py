#!/usr/bin/env python3

'''Read CSV file and transfer it into DataFrame'''

import pandas as pd

data_frame = pd.read_csv('data.csv')

print('CSV file read and transferred to a DataFrame: \n', data_frame)
