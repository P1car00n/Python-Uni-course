import os
import pandas as pd

directory_path = os.fsencode(input('Directory: '))

for file in os.scandir(directory_path):
    df = pd.read_csv(file.path.decode('ascii'))
    df.to_json(file.path.decode('ascii').removesuffix('.csv') + '.json')
    df.to_xml(file.path.decode('ascii').removesuffix('.csv') + '.xml')
