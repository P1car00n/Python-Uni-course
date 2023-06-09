import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def getWebsitesXy(test_size=0.2):
    # TODO: input(('Path to the train dataset: '))
    # Since the decision trees implemented in scikit-learn use only numerical
    # features and these features are interpreted always as continuous numeric
    # variables, we drop them
    dataset = pd.read_csv('/home/arthur/Uni/lab5/data/dataset.csv',
                          date_format='%m/%d/%Y %H:%M').drop(['CHARSET',
                                                              'URL',
                                                              'SERVER',
                                                              'WHOIS_COUNTRY',
                                                              'WHOIS_STATEPRO'],
                                                             axis=1)

    dataset_train, dataset_test = train_test_split(
        dataset, test_size=test_size)

    X_train = dataset_train
    # here I'm setting which column I want to have predicted
    y_train = dataset_train['Type']
    X_test = dataset_test
    y_test = dataset_test['Type']
    return X_train, X_test, y_train, y_test


def getNumbersXy(test_size=0.2):
    np.random.seed(0)
    X = np.random.randn(300, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
