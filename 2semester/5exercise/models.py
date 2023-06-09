from sklearn.metrics import confusion_matrix, r2_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

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
                verbose=2).fit(
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
            train_proba = model.get_prediction_proba(X_train[:2, :])
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
        description='A decision tree classifer for the websites dataset',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=3,
        max_features='sqrt')
    y_pred_dtc_websites = dtc_websites.get_prediction(X_test)

    # Grid search

    printAccuracy(
        models=(
            dtc_websites,
            mnb_circle),
        predictions=(
            y_pred_dtc_websites,
            y_pred_mnb_circle))

    # reinitialize Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getNumbersXy()

    # Numbers
    # Decision tree
    dtc_numbers = DTC(
        X_train,
        y_train,
        description='A decision tree classifer for the random numbers',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=3,
        max_features='sqrt')
    y_pred_dtc_numbers = dtc_numbers.get_prediction(X_test)

    # Grid search

    printAccuracy(
        models=(
            dtc_numbers,
            mnb_circle),
        predictions=(
            y_pred_dtc_numbers,
            y_pred_mnb_circle))

    # grid search for multinomial Vector Classification
    params = {
        'C': (
            1.0, 2.0, 5.0, 10.0), 'degree': (
            3, 6, 9, 12), 'coef0': (
                0.0, 2.0, 5.0, 10.0), 'probability': (
                    False, True)}
    csvc_poly_moon_grid = GridSearcher(
        X_train,
        y_train,
        description='grid search C-Support Multinomial Vector Classification for the moons dataset',
        params=params)
    y_pred_moon_grid = csvc_poly_moon_grid.get_prediction(X_test)
