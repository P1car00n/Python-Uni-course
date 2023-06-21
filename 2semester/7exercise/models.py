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

    def get_score(self, X, y):
        return self.model.score(X, y)

    def get_labels(self):
        return self.model.labels_

    def __repr__(self) -> str:
        return self.description


class KM(Model):

    def __init__(
            self,
            X,
            y,
            description='K-Means clustering',
            **kwargs):
        Model.__init__(self, description,
                       model=KMeans(**kwargs).fit(X, y))


class MBKM(Model):

    def __init__(
            self,
            X,
            y,
            description='Mini-Batch K-Means clustering',
            **kwargs):
        Model.__init__(self, description,
                       model=MiniBatchKMeans(**kwargs).fit(X, y))


if __name__ == '__main__':
    def printAccuracy(models, predictions):
        print('~' * 90)
        for (model, prediction) in zip(models, predictions):

            # Rand index adjusted for chance
            rand = adjusted_rand_score(
                labels_true=y_test, labels_pred=prediction)
            print('Rand index adjusted for chance for',
                  model, 'is as follows: \n', rand)

            # Calinski and Harabasz score
            cnh = calinski_harabasz_score(X=X_train, labels=model.get_labels())
            print('Calinski and Harabasz score for',
                  model, 'is as follows: \n', cnh)

            #  Davies-Bouldin score
            db = davies_bouldin_score(X=X_train, labels=model.get_labels())
            print('Davies-Bouldin score for', model, 'is as follows: \n', db)

    # set Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getBlobsXy()

    # Blobs
    # K-Means clustering
    km_blobs = KM(
        X_train,
        y_train,
        description='a K-Means clustering for blobs',
        n_clusters=2)
    y_pred_km_blobs = km_blobs.get_prediction(X_test)

    # Mini-Batch K-Means clustering
    mbkm_blobs = MBKM(
        X_train,
        y_train,
        description='a Mini-Batch K-Means clustering for blobs',
        n_clusters=2)
    y_pred_mbkm_blobs = mbkm_blobs.get_prediction(X_test)

    printAccuracy(
        models=(
            km_blobs,
            mbkm_blobs
        ),
        predictions=(
            y_pred_km_blobs,
            y_pred_mbkm_blobs
        ))
