from sklearn.datasets import make_blobs, make_circles, make_moons, fetch_covtype


def getBlobsXy():
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [1.5, 0.5]
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers,
                      cluster_std=clusters_std, random_state=0, shuffle=False)
    return X, y


def getCirclesXy():
    X, y = make_circles(500, factor=.1, noise=.1)
    return X, y


def getMoonsXy():
    X, y = make_moons(500, noise=.1)
    return X, y


def getCovtypesXy():
    # TODO: replace with input()
    X, y = fetch_covtype(
        data_home='/home/arthur/Developing/Python-Uni-course/2semester/4exercise/data')
    return X, y
