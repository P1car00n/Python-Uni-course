from sklearn.datasets import make_blobs, make_circles, make_moons, fetch_covtype
from sklearn.model_selection import train_test_split


def getBlobsXy():
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [1.5, 0.5]
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers,
                      cluster_std=clusters_std, random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)
    X_control, X_control_train, y_control, y_control_train = train_test_split(X_train, y_train)
    return X_train, X_test, y_train, y_test, X_control, X_control_train, y_control, y_control_train
