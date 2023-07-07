from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt


n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clustersstd = [1.5, 0.5]
X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers,
                  cluster_std=clustersstd, random_state=0, shuffle=False)

clustering = SpectralClustering().fit(X, y)
fig = plt.figure(figsize=(10, 10))
labels = clustering.labels_
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
