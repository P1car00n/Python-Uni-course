#!/usr/bin/env python3

'''Transfer object Series into index column of the dataframe'''

import pandas as pd
import string

ser = pd.Series([x for x in string.ascii_lowercase])

print(ser.to_frame().reset_index())
