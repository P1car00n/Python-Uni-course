from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd


def getCaliforniaXy(test_size=0.2):
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def getGoogleShareXy():
    # setting thousands since there are numbers like 1,000,111 in the dataset;
    # likewise for dates
    dataset_train = pd.read_csv(
        '/home/arthur/Uni/lab2/data/Google_Stock_Price_Train.csv',
        thousands=',',
        date_format='%m/%d/%Y')  # TODO: input()
    dataset_test = pd.read_csv(
        '/home/arthur/Uni/lab2/data/Google_Stock_Price_Test.csv',
        thousands=',',
        date_format='%m/%d/%Y')

    # here I'm setting what column I want to have predicted
    X_train = dataset_train.drop(['Close', 'Date'], axis=1)
    y_train = dataset_train['Close']
    X_test = dataset_test.drop(['Close', 'Date'], axis=1)
    y_test = dataset_test['Close']
    return X_train, X_test, y_train, y_test

# may be reworked with OOP
