from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

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

    def __repr__(self) -> str:
        return self.description


class MLPC(Model):

    def __init__(
            self,
            X,
            y,
            description='multi-layer Perceptron classifier',
            **kwargs):
        Model.__init__(self, description,
                       model=make_pipeline(
                           MinMaxScaler(),
                           MLPClassifier(early_stopping=True, **kwargs)).fit(X, y))


if __name__ == '__main__':
    def printAccuracy(models, predictions):
        print('~' * 100)
        for (model, prediction) in zip(models, predictions):
            train_E = model.get_prediction(X_train)

            # confusion matrix
            train_CM_acc = confusion_matrix(y_train, train_E)
            test_CM_acc = confusion_matrix(y_test, prediction)
            print(
                'Confusion matrices for',
                model,
                'are \n',
                train_CM_acc,
                'for train data and \n',
                test_CM_acc,
                'for test data')

            # Xs and ys are visible in the scope
            print('Prediction accuracy for', model,
                  'is', model.get_score(X_train, y_train))
            print('Prediction accuracy of y_test for',
                  model, 'is', r2_score(y_test, prediction))

            # recall score
            train_RC_acc = recall_score(y_train, train_E)
            test_RC_acc = recall_score(y_test, prediction)
            print(
                'Recall scores for',
                model,
                'are',
                train_RC_acc,
                'for train data and',
                test_RC_acc,
                'for test data')

            # f1 score
            train_F1_acc = f1_score(y_train, train_E)
            test_F1_acc = f1_score(y_test, prediction)
            print(
                'F1 scores for',
                model,
                'are',
                train_F1_acc,
                'for train data and',
                test_F1_acc,
                'for test data')

            # precision recall curve
            train_PRC_acc = precision_recall_curve(y_train, train_E)
            test_PRC_acc = precision_recall_curve(y_test, prediction)
            print(
                'Precision recall curves for',
                model,
                'are \n',
                train_PRC_acc,
                'for train data and \n',
                test_PRC_acc,
                'for test data')

            # ROC curve
            train_ROC_acc = roc_curve(y_train, train_E)
            test_ROC_acc = roc_curve(y_test, prediction)
            print(
                'ROC curves for',
                model,
                'are \n',
                train_ROC_acc,
                'for train data and \n',
                test_ROC_acc,
                'for test data')

            # ROC AUC
            train_ROCAUC_acc = roc_auc_score(y_train, train_E)
            test_ROCAUC_acc = roc_auc_score(y_test, prediction)
            print(
                'Area Under the Receiver Operating Characteristic Curve for',
                model,
                'are \n',
                train_ROCAUC_acc,
                'for train data and \n',
                test_ROCAUC_acc,
                'for test data')

            # predict_proba
            train_proba = model.get_prediction_proba(X_train)
            print(
                'Posterior probability estimates for',
                model,
                'are as follows: \n',
                train_proba)

    # set Xs and ys
    X_train, X_test, y_train, y_test, X_control, X_control_train, y_control, y_control_train = data_provider.getBlobsXy()

    # Blobs
    # Multi-layer Perceptron classifier
    mplc100_blobs = MLPC(
        X_train,
        y_train,
        description='multi-layer Perceptron classifier for blobs',
        hidden_layer_sizes=(100,))
    y_pred_mplc100_blobs = mplc100_blobs.get_prediction(X_test)

    mplc300_blobs = MLPC(
        X_train,
        y_train,
        description='multi-layer Perceptron classifier for blobs',
        hidden_layer_sizes=(300,))
    y_pred_mplc300_blobs = mplc300_blobs.get_prediction(X_test)

    mplc500_blobs = MLPC(
        X_train,
        y_train,
        description='multi-layer Perceptron classifier for blobs',
        hidden_layer_sizes=(500,))
    y_pred_mplc500_blobs = mplc500_blobs.get_prediction(X_test)

    printAccuracy(
        models=(
            mplc100_blobs,
            mplc300_blobs,
            mplc500_blobs),
        predictions=(
            y_pred_mplc100_blobs,
            y_pred_mplc300_blobs,
            y_pred_mplc500_blobs))


    ## Ensembles Voting
    #clf1 = MLPClassifier(hidden_layer_sizes=(100,))
    #clf2 = MLPClassifier(hidden_layer_sizes=(300,))
    #clf3 = MLPClassifier(hidden_layer_sizes=(500,))
    #voting_clf_hard = VotingClassifier(
    #    estimators=[
    #        ('clf1',
    #         clf1),
    #        ('clf2',
    #         clf2),
    #        ('clf3',
    #         clf3)],
    #    voting='hard')
    #voting_clf_hard.fit(X_train, y_train)
    #y_pred_hard = voting_clf_hard.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred_hard)
#
    ## Plotting decision boundaries
    #x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    #y_min, y_max = y_train[:, 1].min() - 1, y_train[:, 1].max() + 1
    #xx, yy = np.meshgrid(
    #    np.arange(
    #        x_min, x_max, 0.1), np.arange(
    #        y_min, y_max, 0.1))
    #f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
#
    #for idx, clf, tt in zip(product([0, 1], [0, 1]), [clf1, clf2, clf3, voting_clf_hard], [
    #                        'Multi-layer Perceptron, Hidden layer size -- 100', 'Multi-layer Perceptron, Hidden layer size -- 300', 'Multi-layer Perceptron, Hidden layer size -- 500', 'Hard Voting']):
    #    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #    Z = Z.reshape(xx.shape)
    #    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    #    axarr[idx[0], idx[1]].scatter(
    #        X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
    #    axarr[idx[0], idx[1]].set_title(tt)
#
    #plt.show()
#