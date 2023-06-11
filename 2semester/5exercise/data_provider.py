import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def getWebsitesXy(test_size=0.2):
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
        dataset[column] = LabelEncoder().fit_transform(
            dataset[column].astype(str))

    dataset_train, dataset_test = train_test_split(
        dataset, test_size=test_size)

    X_train = dataset_train.drop('Type', axis=1)
    # here I'm setting which column I want to have predicted
    y_train = dataset_train['Type']
    X_test = dataset_test.drop('Type', axis=1)
    y_test = dataset_test['Type']
    return X_train, X_test, y_train, y_test


def getNumbersXy(test_size=0.2):
    np.random.seed(0)
    X = np.random.randn(300, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
