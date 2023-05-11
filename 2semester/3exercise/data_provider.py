from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split


def getMoonsXy(test_size=0.2):
    X, y = make_moons()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def getDigitsXy(test_size=0.2):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
