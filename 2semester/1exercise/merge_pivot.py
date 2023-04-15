import os
import pandas as pd

input_directory_path = os.fsencode(input('Input directory: '))
output_directory_path = input('Output directory: ') + '/'

dfs = []

for file in os.scandir(input_directory_path):
    file_path = file.path.decode('ascii')
    if file_path.endswith('.json'):
        dfs.append(pd.read_json(file_path))

df = dfs[0]
df_c = pd.concat(dfs)

for i in range(1, len(dfs)):
    df.merge(dfs[i])
df.to_xml(output_directory_path + 'merged' + '.xml')

df_c.to_xml(output_directory_path + 'concatenated' + '.xml')

df.pivot(
    columns=(
        'Penalty',
        'Cost',
        'Presentation',
        'Design',
        'Accel',
        'Skid_pad',
        'Autocross',
        'Endurance',
        'Efficiency',
        'Total')).to_xml(
            output_directory_path +
            'pivoted' +
    '.xml')
