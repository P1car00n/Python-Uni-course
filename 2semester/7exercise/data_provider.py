import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA


def getBlobsX(plot=False):
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [2.0, 1.0]
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers,
                      cluster_std=clusters_std, random_state=0, shuffle=False)

    if plot:
        make_plot(X, y=y)
        return

    return X, y


def getClusterX(plot=False):
    np.random.seed(0)
    n_points_per_cluster = 30000
    C1 = [-6, -2] + 0.7 * np.random.randn(n_points_per_cluster, 2)
    C2 = [-2, 2] + 0.3 * np.random.randn(n_points_per_cluster, 2)
    C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
    C4 = [4, 4] + 0.1 * np.random.randn(n_points_per_cluster, 2)
    C5 = [5, 0] + 1.4 * np.random.randn(n_points_per_cluster, 2)
    C6 = [5, 6] + 2.0 * np.random.randn(n_points_per_cluster, 2)
    X = np.vstack((C1, C2, C3, C4, C5, C6))

    y = np.concatenate((
        np.zeros(n_points_per_cluster),
        np.ones(n_points_per_cluster),
        2 * np.ones(n_points_per_cluster),
        3 * np.ones(n_points_per_cluster),
        4 * np.ones(n_points_per_cluster),
        5 * np.ones(n_points_per_cluster)
    ))

    # y = np.column_stack((X, labels))

    # using it istead of y, as there are no labels
    # spectral1 = SpectralClustering(
    #    n_clusters=6,
    #    eigen_solver='amg',
    #    affinity="nearest_neighbors",
    #    n_jobs=-1).fit_predict(X)

    if plot:
        make_plot(X, y)
        return

    return X, y


def make_plot(X, y):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X)
    plt.scatter(proj[:, 0], proj[:, 1], c=y, cmap="Paired")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    choice = input('1 for blobs, 2 for clusters: ')
    if choice == '1':
        getBlobsX(plot=True)
    elif choice == '2':
        getClusterX(plot=True)
