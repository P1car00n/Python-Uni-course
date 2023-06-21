from sklearn.cluster import KMeans, MiniBatchKMeans
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
        Model.__init__(self, description,
                       model=KMeans(**kwargs).fit(X))


class MBKM(Model):

    def __init__(
            self,
            X,
            description='Mini-Batch K-Means clustering',
            **kwargs):
        Model.__init__(self, description,
                       model=MiniBatchKMeans(**kwargs).fit(X))


if __name__ == '__main__':
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

    # set Xs and ys
    X, y = data_provider.getBlobsX()

    # Blobs
    # K-Means clustering
    km_blobs = KM(
        X,
        description='a K-Means clustering for blobs',
        n_clusters=2)
    pred_km_blobs = km_blobs.get_prediction(X)

    # Mini-Batch K-Means clustering
    mbkm_blobs = MBKM(
        X,
        description='a Mini-Batch K-Means clustering for blobs',
        n_clusters=2)
    pred_mbkm_blobs = mbkm_blobs.get_prediction(X)

    printAccuracy(
        models=(
            km_blobs,
            mbkm_blobs
        ),
        predictions=(
            pred_km_blobs,
            pred_mbkm_blobs
        ))

    # reset Xs and ys
    X, y = data_provider.getClusterX()

    # Clusters
    # K-Means clustering
    km_clusters = KM(
        X,
        description='a K-Means clustering for clusters',
        n_clusters=6)
    pred_km_clusters = km_clusters.get_prediction(X)

    # Mini-Batch K-Means clustering
    mbkm_clusters = MBKM(
        X,
        description='a Mini-Batch K-Means clustering for clusters',
        n_clusters=6)
    pred_mbkm_clusters = mbkm_clusters.get_prediction(X)

    printAccuracy(
        models=(
            km_clusters,
            mbkm_clusters
        ),
        predictions=(
            pred_km_clusters,
            pred_mbkm_clusters
        ))
