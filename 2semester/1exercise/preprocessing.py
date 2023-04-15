import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_directory_path = os.fsencode(input('Input directory: '))
output_directory_path = input('Output directory: ') + '/'

for file in os.scandir(input_directory_path):
    file_path = file.path.decode('ascii')
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)

        df.fillna(0)

        df[['Penalty',
            'Cost',
            'Presentation',
            'Design',
            'Accel',
            'Skid_pad',
            'Autocross',
            'Endurance',
            'Efficiency',
            'Total']] = MinMaxScaler().fit_transform(
            df[
                [
                    'Penalty',
                    'Cost',
                    'Presentation',
                    'Design',
                    'Accel',
                    'Skid_pad',
                    'Autocross',
                    'Endurance',
                    'Efficiency',
                    'Total']])

        pd.DataFrame(
            df,
            columns=df.columns).to_json(
            output_directory_path +
            file.name.decode('ascii').removesuffix('.csv') +
            '.json')
        pd.DataFrame(
            df,
            columns=df.columns).to_xml(
            output_directory_path +
            file.name.decode('ascii').removesuffix('.csv') +
            '.xml')
