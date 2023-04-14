#!/usr/bin/env python3

'''Create two data frames using the two Dicts, Merge two
data frames, and append the second data frame as a new
column to the first data frame.'''

import pandas as pd

dict1 = {
    'Number': [
        34, 23, 65, 55], 'Fruit': [
            'Apple', 'Banana', 'Pear', 'Tomato']}
dict2 = {
    'Number': [
        87,
        68,
        43,
        23],
    'Fruit': [
        'Watermelon',
        'Mango',
        'Strawberry',
        'Blackberry']}
data_frame1 = pd.DataFrame.from_dict(dict1)
data_frame2 = pd.DataFrame.from_dict(dict2)
print('First DataFrame: \n', data_frame1)
print('Second DataFrame: \n', data_frame2)

data_frame_merged = pd.concat(
    [data_frame1, data_frame2]).reset_index(drop=True)
print('Merged frames: \n', data_frame_merged)

data_frame_appended = pd.concat([data_frame1, data_frame2], axis=1)
print('Appended frame: \n', data_frame_appended)
