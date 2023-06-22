from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score)

import data_provider


class Model:

    def __init__(self, description, model):
        self.description = description
        self.model = model

    def get_prediction(self, samples):
        return self.model.predict(samples)

    def get_prediction_proba(self, samples):
        return self.model.predict_proba(samples)

    def get_score(self, X):
        return self.model.score(X)

    def get_labels(self):
        return self.model.labels_

    def __repr__(self) -> str:
        return self.description


class KM(Model):

    def __init__(
            self,
            X,
            description='K-Means clustering',
            **kwargs):
        Model.__init__(
            self,
            description,
            model=KMeans(
                max_iter=600,
                n_init='auto',
                **kwargs).fit(X))


class MBKM(Model):

    def __init__(
            self,
            X,
            description='Mini-Batch K-Means clustering',
            **kwargs):
        Model.__init__(self, description,
                       model=MiniBatchKMeans(**kwargs).fit(X))


if __name__ == '__main__':
    def visualize(X, y):
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)
        plt.figure(1)
        plt.scatter(proj[:, 0], proj[:, 1], c=y)
        plt.show()

    def printAccuracy(models, predictions):
        print('~' * 90)
        for (model, prediction) in zip(models, predictions):

            # Rand index adjusted for chance
            rand = adjusted_rand_score(
                labels_true=y, labels_pred=prediction)
            print('Rand index adjusted for chance for',
                  model, 'is as follows: \n', rand)

            # Calinski and Harabasz score
            cnh = calinski_harabasz_score(
                X=X, labels=prediction)  # model.get_labels()
            print('Calinski and Harabasz score for',
                  model, 'is as follows: \n', cnh)

            #  Davies-Bouldin score
            db = davies_bouldin_score(X, labels=prediction)
            print('Davies-Bouldin score for', model, 'is as follows: \n', db)

            # visualizing
            visualize(X, prediction)

    # set Xs and ys
    X, y = data_provider.getBlobsX()

    # Blobs
    # K-Means clustering
    km_blobs = KM(
        X,
        description='a K-Means clustering for blobs with lloyd',
        n_clusters=2)
    pred_km_blobs = km_blobs.get_prediction(X)

    km_blobs1 = KM(
        X,
        description='a K-Means clustering for blobs with elkan',
        n_clusters=2,
        algorithm='elkan')
    pred_km_blobs1 = km_blobs1.get_prediction(X)

    km_blobs_rand = KM(
        X,
        description='a K-Means clustering for blobs with lloyd and random initialization',
        n_clusters=2,
        init='random')
    pred_km_blobs_rand = km_blobs_rand.get_prediction(X)

    km_blobs1_rand = KM(
        X,
        description='a K-Means clustering for blobs with elkan and random initialization',
        n_clusters=2,
        algorithm='elkan',
        init='random')
    pred_km_blobs1_rand = km_blobs1_rand.get_prediction(X)

    # Mini-Batch K-Means clustering
    mbkm_blobs = MBKM(
        X,
        description='a Mini-Batch K-Means clustering for blobs',
        n_clusters=2)
    pred_mbkm_blobs = mbkm_blobs.get_prediction(X)

    printAccuracy(
        models=(
            km_blobs,
            km_blobs1,
            km_blobs_rand,
            km_blobs1_rand,
            mbkm_blobs
        ),
        predictions=(
            pred_km_blobs,
            pred_km_blobs1,
            pred_km_blobs_rand,
            pred_km_blobs1_rand,
            pred_mbkm_blobs
        ))

    # reset Xs and ys
    X, y = data_provider.getClusterX()

    # Clusters
    # K-Means clustering
    km_clusters = KM(
        X,
        description='a K-Means clustering for clusters with lloyd',
        n_clusters=6)
    pred_km_clusters = km_clusters.get_prediction(X)

    km_clusters1 = KM(
        X,
        description='a K-Means clustering for clusters with elkan',
        n_clusters=2,
        algorithm='elkan')
    pred_km_clusters1 = km_clusters1.get_prediction(X)

    km_clusters_rand = KM(
        X,
        description='a K-Means clustering for clusters with lloyd and random initialization',
        n_clusters=2,
        init='random')
    pred_km_clusters_rand = km_clusters_rand.get_prediction(X)

    km_clusters1_rand = KM(
        X,
        description='a K-Means clustering for clusters with elkan and random initialization',
        n_clusters=2,
        algorithm='elkan',
        init='random')
    pred_km_clusters1_rand = km_clusters1_rand.get_prediction(X)

    # Mini-Batch K-Means clustering
    mbkm_clusters = MBKM(
        X,
        description='a Mini-Batch K-Means clustering for clusters',
        n_clusters=6)
    pred_mbkm_clusters = mbkm_clusters.get_prediction(X)

    printAccuracy(
        models=(
            km_clusters,
            km_clusters1,
            km_clusters_rand,
            km_clusters1_rand,
            mbkm_clusters
        ),
        predictions=(
            pred_km_clusters,
            pred_km_clusters1,
            pred_km_clusters_rand,
            pred_km_clusters1_rand,
            pred_mbkm_clusters
        ))
