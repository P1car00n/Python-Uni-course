import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv(
    input(
        ('Path to the train dataset: ')),
    date_format='%d/%m/%Y %H:%M',
    parse_dates=True).fillna(0)
for column in [
    'CHARSET',
    'URL',
    'SERVER',
    'WHOIS_COUNTRY',
    'WHOIS_STATEPRO',
    'WHOIS_REGDATE',
        'WHOIS_UPDATED_DATE']:
    dataset[column] = LabelEncoder().fit_transform(dataset[column].astype(str))
X = dataset.drop('Type', axis=1)
y = dataset['Type']

# change which columns to plot here
X = X[['SERVER', 'WHOIS_STATEPRO']]

model = DecisionTreeClassifier()
model.fit(X, y)

disp = DecisionBoundaryDisplay.from_estimator(
    model, X, response_method="predict")

plt.figure(figsize=(8, 6))
disp.plot(ax=plt.gca(), alpha=0.5)
plt.show()
