import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import data_provider


def get_prediction(model, samples):
    return model.predict(samples)


class Model:

    def __init__(self, description, model):
        self.description = description
        self.model = model

    def get_prediction(self, samples):
        return self.model.predict(samples)

    def get_score(self, X, y):
        return self.model.score(X, y)

    def __repr__(self) -> str:
        return self.description


class LRM(Model):

    def __init__(self, X, y, description='Linear regression model', **kwargs):
        # **kwargs is for fit_intercept=True
        Model.__init__(self, description,
                       model=LinearRegression(**kwargs).fit(X, y))


if __name__ == '__main__':
    # set Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getCaliforniaXy()

    # houses
    lrm_cal_intercept = LRM(
        X_train,
        y_train,
        description='linear regression model with intercept for California housing prices',
        fit_intercept=True)
    lrm_cal_no_intercept = LRM(
        X_train,
        y_train,
        description='linear regression model with no intercept for California housing prices',
        fit_intercept=False)
    y_pred_lrm_cal_intercept = lrm_cal_intercept.get_prediction(X_test)
    y_pred_lrm_cal_no_intercept = lrm_cal_no_intercept.get_prediction(X_test)
    print("Prediction accuracy (R^2) for", lrm_cal_intercept,
          'is', lrm_cal_intercept.get_score(X_train, y_train))
    print("Prediction accuracy (R^2) for", lrm_cal_no_intercept,
          'is', lrm_cal_no_intercept.get_score(X_train, y_train))
    print(
        "Prediction accuracy (R^2) of y_test for",
        lrm_cal_intercept,
        'is',
        r2_score(
            y_test,
            y_pred_lrm_cal_intercept))
    print(
        "Prediction accuracy (R^2) of y_test for",
        lrm_cal_no_intercept,
        'is',
        r2_score(
            y_test,
            y_pred_lrm_cal_no_intercept))

    # reinitialize Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getGoogleShareXy()

    # google shares
    lrm_gog_intercept = LRM(
        X_train,
        y_train,
        description='linear regression model with intercept for Google share prices',
        fit_intercept=True)
    lrm_gog_no_intercept = LRM(
        X_train,
        y_train,
        description='linear regression model with no intercept for Google share prices',
        fit_intercept=False)
    y_pred_lrm_gog_intercept = lrm_gog_intercept.get_prediction(X_test)
    y_pred_lrm_gog_no_intercept = lrm_gog_no_intercept.get_prediction(
        X_test)
    print("Prediction accuracy (R^2) for", lrm_gog_intercept,
          'is', lrm_gog_intercept.get_score(X_train, y_train))
    print(
        "Prediction accuracy (R^2) for",
        lrm_gog_no_intercept,
        'is',
        lrm_gog_no_intercept.get_score(
            X_train,
            y_train))
    print(
        "Prediction accuracy (R^2) of y_test for",
        lrm_gog_intercept,
        'is',
        r2_score(
            y_test,
            y_pred_lrm_gog_intercept))
    print(
        "Prediction accuracy (R^2) of y_test for",
        lrm_gog_intercept,
        'is',
        r2_score(
            y_test,
            y_pred_lrm_gog_no_intercept))
