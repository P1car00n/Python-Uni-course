import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/home/arthur/Uni/lab1/data/preprocessing/all_completed_laps_diff.csv')

df.fillna(0)

df_scaled = MinMaxScaler().fit_transform(df.to_numpy())