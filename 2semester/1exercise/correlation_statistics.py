import pandas as pd
from fpdf import FPDF

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

column = input('Column to compute: ')
save_to_path = input('Path to save output to: ')

cor1 = df['Penalty'].corr(df['Total'])
cor2 = df['Cost'].corr(df['Efficiency'])
median = df[f'{column}'].median()
mean = df[f'{column}'].mean()
description = df[f'{column}'].describe()


with open(save_to_path + '/analysis.txt', encoding="utf-8", mode='w') as f:
    f.write(
        'Correlation between the columns "Penalty" and "Total": ' +
        str(cor1) +
        '\n')
    f.write(
        'Correlation between the columns "Cost" and "Efficiency": ' +
        str(cor2) +
        '\n')
    f.write(f'The median of the "{column}" column: ' + str(median) + '\n')
    f.write(f'The mean of the "{column}" column: ' + str(mean) + '\n')
    f.write(
        f'An in-depth description of the "{column}" column:\n' +
        str(description) +
        '\n')

pdf = FPDF()
pdf.add_page()
pdf.set_font('helvetica', 'B', 16)
pdf.write(
    5,
    'Correlation between the columns "Penalty" and "Total": ' +
    str(cor1) +
    '\n')
pdf.write(
    5,
    'Correlation between the columns "Cost" and "Efficiency": ' +
    str(cor2) +
    '\n')
pdf.write(5, f'The median of the "{column}" column: ' + str(median) + '\n')
pdf.write(5, f'The mean of the "{column}" column: ' + str(mean) + '\n')
pdf.write(
    5,
    f'An in-depth description of the "{column}" column:\n' +
    str(description) +
    '\n')
pdf.output(save_to_path + '/analysis.pdf')
