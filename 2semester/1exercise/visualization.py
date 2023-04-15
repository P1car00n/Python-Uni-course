import pandas as pd
import matplotlib.pyplot as plt


file_path = input('File path: ')

if file_path.endswith('.xml'):
    df = pd.read_xml(file_path)
elif file_path.endswith('.json'):
    df = pd.read_json(file_path)
elif file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
else:
    print('Unknown file format')
    exit()

print('Histogram for the "Total" column')
f1 = plt.figure(1)
df.hist(column='Total')
plt.show()

print('Boxplot for the "Autocross", "Endurance" and "Efficiency" columns')
f2 = plt.figure(2)
df.boxplot(column=['Autocross', 'Endurance', 'Efficiency'])
plt.show()

print('Area plot for the whole dataset')
f3 = plt.figure(3)
df.plot.area()
plt.show()
