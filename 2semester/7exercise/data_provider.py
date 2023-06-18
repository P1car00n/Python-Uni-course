import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering


def getBlobsXy(test_size=0.2, plot=False):
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [2.0, 1.0]
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers,
                      cluster_std=clusters_std, random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    
    if plot:
        make_plot(X, y=y)
        return
    
    return X_train, X_test, y_train, y_test


def getClusterX(plot=False):
    np.random. seed(0)
    n_points_per_cluster = 30000
    C1 = [-6, -2] + 0.7 * np.random.randn(n_points_per_cluster, 2)
    C2 = [-2, 2] + 0.3 * np.random.randn(n_points_per_cluster, 2)
    C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
    C4 = [4, 4] + 0.1 * np.random.randn(n_points_per_cluster, 2)
    C5 = [5, 0] + 1.4 * np.random.randn(n_points_per_cluster, 2)
    C6 = [5, 6] + 2.0 * np.random.randn(n_points_per_cluster, 2)
    X = np.vstack((C1, C2, C3, C4, C5, C6))

    if plot:
        # takes too long, can't proceed
        spectral1 = SpectralClustering(n_clusters=6, affinity="nearest_neighbors", n_jobs=-1).fit_predict(X)
        make_plot(X, spectral1=spectral1)
        return

    return X

def make_plot(X, y=None, spectral1=None):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X)
    if spectral1 is not None and y is None:
        plt.scatter(proj[:, 0], proj[:, 1], c = spectral1, cmap="Paired")
    elif y is not None and spectral1 is None:
        plt.scatter(proj[:, 0], proj[:, 1], c = y, cmap="Paired")
    else:
        return
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    choice = input('1 for blobs, 2 for clusters: ')
    if choice == '1':
        getBlobsXy(plot=True)
    elif choice == '2':
        getClusterX(plot=True)
