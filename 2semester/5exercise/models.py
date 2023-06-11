from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

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


class DTC(Model):

    def __init__(
            self,
            X,
            y,
            description='Decision Tree model',
            **kwargs):
        Model.__init__(self, description,
                       model=DecisionTreeClassifier(**kwargs).fit(X, y))


class GridSearcher(Model):
    def __init__(
            self,
            X,
            y,
            description='Grid search',
            params={}):
        Model.__init__(
            self,
            description,
            model=GridSearchCV(
                DecisionTreeClassifier(),
                param_grid=params,
                n_jobs=-1,
                verbose=1).fit(
                X,
                y))


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
            # can't be checked, as there are bool values in y_pred
            # print('Prediction accuracy of y_test for',
            #      model, 'is', r2_score(y_test, prediction))

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
    X_train, X_test, y_train, y_test = data_provider.getWebsitesXy()

    # Websites
    # Decision tree
    dtc_websites = DTC(
        X_train,
        y_train,
        description='a decision tree classifier for the websites dataset',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=3,
        max_features='sqrt')
    y_pred_dtc_websites = dtc_websites.get_prediction(X_test)

    # Grid search
    # setting up the parameters
    params = {
        'min_samples_split': (
            2, 4, 6), 'min_samples_leaf': (
            1, 3, 6), 'max_leaf_nodes': (
                3, 6, None), 'max_features': (
                    'sqrt', 'log2', None)}

    dtc_websites_grid = GridSearcher(
        X_train,
        y_train,
        description='A grid search decision tree classifer for the websites dataset',
        params=params)
    y_pred_dtc_websites_grid = dtc_websites_grid.get_prediction(X_test)

    printAccuracy(
        models=(
            dtc_websites,
            dtc_websites_grid),
        predictions=(
            y_pred_dtc_websites,
            y_pred_dtc_websites_grid))

    # reinitialize Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getNumbersXy()

    # Numbers
    # Decision tree
    dtc_numbers = DTC(
        X_train,
        y_train,
        description='a decision tree classifier for the random numbers',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=3,
        max_features='sqrt')
    y_pred_dtc_numbers = dtc_numbers.get_prediction(X_test)

    # Grid search
    # reusing the parameters

    dtc_numbers_grid = GridSearcher(
        X_train,
        y_train,
        description='A grid search decision tree classifer for the numbers dataset',
        params=params)
    y_pred_dtc_numbers_grid = dtc_numbers_grid.get_prediction(X_test)

    printAccuracy(
        models=(
            dtc_numbers,
            dtc_numbers_grid),
        predictions=(
            y_pred_dtc_numbers,
            y_pred_dtc_numbers_grid))
