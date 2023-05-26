from sklearn.metrics import confusion_matrix, r2_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC
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


class GNB(Model):

    def __init__(
            self,
            X,
            y,
            description='Gaussian Naive Bayes model',
            **kwargs):
        Model.__init__(self, description,
                       model=GaussianNB(**kwargs).fit(X, y))


class MNB(Model):

    def __init__(
            self,
            X,
            y,
            description='Naive Bayes classifier for multinomial models',
            **kwargs):
        Model.__init__(self, description,
                       model=make_pipeline(
                           MinMaxScaler(),
                           MultinomialNB(**kwargs)).fit(X, y))


class CSVC(Model):

    def __init__(
            self,
            X,
            y,
            description='C-Support Vector Classifier',
            **kwargs):
        Model.__init__(self, description,
                       model=make_pipeline(
                           MinMaxScaler(),
                           SVC(**kwargs)).fit(X, y))


class LSVC(Model):

    def __init__(
            self,
            X,
            y,
            description='Linear Support Vector Classifier',
            **kwargs):
        Model.__init__(self, description,
                       model=make_pipeline(
                           MinMaxScaler(),
                           LinearSVC(**kwargs)).fit(X, y))


# class GridSearcher(Model):
#    def __init__(
#            self,
#            X,
#            y,
#            description='Grid search',
#            params={}):
#        Model.__init__(
#            self,
#            description,
#            model=GridSearchCV(
#                LogisticRegression(
#                    solver='lbfgs',
#                    multi_class='multinomial',
#                    n_jobs=-1),
#                param_grid=params,
#                n_jobs=-1,
#                verbose=2).fit(
#                X,
#                y))

if __name__ == '__main__':
    def printAccuracyBayes(models, predictions):
        print('~' * 100)
        for (model, prediction) in zip(models, predictions):

            # predict_proba
            train_proba = model.get_prediction_proba(X_train[:2, :])
            print(
                'Posterior probability estimates for',
                model,
                'are as follows: \n',
                train_proba)

    def printAccuracySVN(models, predictions):
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
            print('Prediction accuracy (for', model,
                  'is', model.get_score(X_train, y_train))
            print('Prediction accuracy (R^2) of y_test for',
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

    # set Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getBlobsXy()

    # Naive Bayes
    # Blobs
    gnb_blob = GNB(
        X_train,
        y_train,
        description='Gaussian Naive Bayes for the blobs dataset')
    y_pred_gnb_blob = gnb_blob.get_prediction(X_test)

    mnb_blob = MNB(
        X_train,
        y_train,
        description='Multinomial Naive Bayes classifier for the blobs dataset')
    y_pred_mnb_blob = mnb_blob.get_prediction(X_test)
    printAccuracyBayes(
        models=(
            gnb_blob,
            mnb_blob),
        predictions=(
            y_pred_gnb_blob,
            y_pred_mnb_blob))

    # reinitialize Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getCirclesXy()

    # Circles
    gnb_circle = GNB(
        X_train,
        y_train,
        description='Gaussian Naive Bayes for the circle dataset')
    y_pred_gnb_circle = gnb_circle.get_prediction(X_test)

    mnb_circle = MNB(
        X_train,
        y_train,
        description='Multinomial Naive Bayes classifier for the circle dataset')
    y_pred_mnb_circle = mnb_circle.get_prediction(X_test)
    printAccuracyBayes(
        models=(
            gnb_circle,
            mnb_circle),
        predictions=(
            y_pred_gnb_circle,
            y_pred_mnb_circle))

    # reinitialize Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getMoonsXy()

    # SVM
    # Moons
    lsvc_moon = LSVC(
        X_train,
        y_train,
        description='Linear Support Vector Classification for the moons dataset',
        C=1.0)
    y_pred_lsvc_moon = lsvc_moon.get_prediction(X_test)

    csvc_linear_moon = CSVC(
        X_train,
        y_train,
        description='C-Support Linear Vector Classification for the moons dataset',
        C=1.0,
        kernel='linear')
    y_pred_csvc_linear_moon = csvc_linear_moon.get_prediction(X_test)

    csvc_poly_moon = CSVC(
        X_train,
        y_train,
        description='C-Support Multinomial Vector Classification for the moons dataset',
        C=1.0,
        kernel='poly')
    y_pred_csvc_poly_moon = csvc_poly_moon.get_prediction(X_test)

    printAccuracySVN(
        models=(
            lsvc_moon,
            csvc_linear_moon,
            csvc_poly_moon),
        predictions=(
            y_pred_lsvc_moon,
            y_pred_csvc_linear_moon,
            y_pred_csvc_poly_moon))

    # reinitialize Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getCovtypesXy()

    # Covertypes
    lsvc_covtype = LSVC(
        X_train,
        y_train,
        description='Linear Support Vector Classification for the covertypes dataset',
        C=1.0)
    y_pred_lsvc_covtype = lsvc_covtype.get_prediction(X_test)

    csvc_covtype = CSVC(
        X_train,
        y_train,
        description='C-Support Vector Classification for the covertypes dataset',
        C=1.0,
        kernel='linear')
    y_pred_csvc_covtype = csvc_covtype.get_prediction(X_test)

    csvc_poly_covtype = CSVC(
        X_train,
        y_train,
        description='C-Support Multinomial Vector Classification for the covertypes dataset',
        C=1.0,
        kernel='poly')
    y_pred_csvc_poly_covtype = csvc_poly_covtype.get_prediction(X_test)

    printAccuracySVN(
        models=(
            lsvc_covtype,
            csvc_covtype,
            csvc_poly_covtype),
        predictions=(
            y_pred_lsvc_covtype,
            y_pred_csvc_covtype,
            y_pred_csvc_poly_covtype))

#
    # grid search for multinomial logistic regression
    # as above --> params = {'max_iter': (100, 200, 500, 1000)}
#
    # lgm_digit_grid = GridSearcher(
    #    X_train,
    #    y_train,
    #    description='grid search logistic regression model with multinomial solver for the digits dataset',
    #    params=params)
    # y_pred_digit_grid = lgm_digit_grid.get_prediction(X_test)
    # printAccuracy(
    #    models=(
    #        lgm_digit_grid,),
    #    predictions=(
    #        y_pred_digit_grid,))
