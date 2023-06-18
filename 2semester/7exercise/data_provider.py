import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def getBlobsXy(test_size=0.2):
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [2.0, 1.0]
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers,
                      cluster_std=clusters_std, random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def getClusterXy(test_zise=0.2):
    np.random. seed(0)
    n_points_per_cluster = 30000
    C1 = [-6, -2] + 0.7 * np.random.randn(n_points_per_cluster, 2)
    C2 = [-2, 2] + 0.3 * np.random.randn(n_points_per_cluster, 2)
    C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
    C4 = [4, 4] + 0.1 * np.random.randn(n_points_per_cluster, 2)
    C5 = [5, 0] + 1.4 * np.random.randn(n_points_per_cluster, 2)
    C6 = [5, 6] + 2.0 * np.random.randn(n_points_per_cluster, 2)
    X = np.vstack((C1, C2, C3, C4, C5, C6))
