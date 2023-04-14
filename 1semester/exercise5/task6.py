#!/usr/bin/env python3

'''Delete upper and lowe 5% in object DataFrame'''

import pandas as pd


def get_five_percent(number):
    return number * 0.05


data_frame = pd.read_csv('data.csv')
print('Original data: \n', data_frame)

frame_length = len(data_frame)

to_delete = int(get_five_percent(frame_length))
data_frame.drop([x for x in range(to_delete)] +
                [x for x in range(frame_length -
                                  1, frame_length -
                                  to_delete -
                                  1, -
                                  1)], inplace=True)
print('Data with the upper and lower 5% dropped: \n', data_frame)
