from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def getCaliforniaXy(test_size=0.2):
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
