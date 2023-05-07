from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

import data_provider


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


class RidgeModel(Model):

    def __init__(self, X, y, description='Ridge regression model', **kwargs):
        # **kwargs is for alpha
        Model.__init__(self, description,
                       model=Ridge(**kwargs).fit(X, y))


class LinearPipiliner(Model):
    def __init__(self, X, y, description='Polynomial linear regression model'):
        Model.__init__(
            self,
            description,
            model=make_pipeline(
                PolynomialFeatures(),
                StandardScaler(),
                LinearRegression()).fit(X, y))


class GridSearcher(Model):
    def __init__(
            self,
            X,
            y,
            description='Polynomial linear regression model optimized with grid search',
            params={}):
        pipe = Pipeline([('sc', StandardScaler()), ('pf',
                        PolynomialFeatures()), ('lr', LinearRegression())])
        Model.__init__(
            self,
            description,
            model=GridSearchCV(
                pipe,
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

            # MAE
            train_MAE_acc = mean_absolute_error(y_train, train_E)
            test_MAE_acc = mean_absolute_error(y_test, prediction)
            print(
                'Mean absolute errors for',
                model,
                'are',
                train_MAE_acc,
                'for train data and',
                test_MAE_acc,
                'for test data')

            # MAPE
            train_MAPE_acc = mean_absolute_percentage_error(y_train, train_E)
            test_MAPE_acc = mean_absolute_percentage_error(y_test, prediction)
            print(
                'Mean absolute percentage errors for',
                model,
                'are',
                train_MAPE_acc,
                'for train data and',
                test_MAPE_acc,
                'for test data')

            # RMSE
            train_RMSE_acc = mean_squared_error(y_train, train_E)
            test_RMSE_acc = mean_squared_error(y_test, prediction)
            print(
                'Mean squared errors for',
                model,
                'are',
                train_RMSE_acc,
                'for train data and',
                test_RMSE_acc,
                'for test data')

            # Xs and ys are visible in the scope
            print('Prediction accuracy (R^2) for', model,
                  'is', model.get_score(X_train, y_train))
            print('Prediction accuracy (R^2) of y_test for',
                  model, 'is', r2_score(y_test, prediction))

    # set Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getCaliforniaXy()

    # houses
    # Linear Regression
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
    printAccuracy(
        models=(
            lrm_cal_intercept,
            lrm_cal_no_intercept),
        predictions=(
            y_pred_lrm_cal_intercept,
            y_pred_lrm_cal_no_intercept))

    # Ridge regression
    rdg_cal_alpha1 = RidgeModel(
        X_train,
        y_train,
        description='ridge regression model with alpha = 1.0 for California housing prices',
        alpha=1.0)
    rdg_cal_alpha10 = RidgeModel(
        X_train,
        y_train,
        description='ridge regression model with alpha = 10.0 for California housing prices',
        alpha=10.0)
    rdg_cal_alpha100 = RidgeModel(
        X_train,
        y_train,
        description='ridge regression model with alpha = 100.0 for California housing prices',
        alpha=100.0)
    y_pred_rdg_cal_alpha1 = rdg_cal_alpha1.get_prediction(X_test)
    y_pred_rdg_cal_alpha10 = rdg_cal_alpha10.get_prediction(X_test)
    y_pred_rdg_cal_alpha100 = rdg_cal_alpha100.get_prediction(X_test)
    printAccuracy(
        models=(
            rdg_cal_alpha1,
            rdg_cal_alpha10,
            rdg_cal_alpha100),
        predictions=(
            y_pred_rdg_cal_alpha1,
            y_pred_rdg_cal_alpha10,
            y_pred_rdg_cal_alpha100))

    # Polynomial stuff + linear regression
    lrm_cal_pol = LinearPipiliner(
        X_train,
        y_train,
        description='linear regression model <<polynomized>> for California housing prices')
    y_pred_cal_pol = lrm_cal_pol.get_prediction(X_test)
    printAccuracy(
        models=(
            lrm_cal_pol,),
        predictions=(
            y_pred_cal_pol,))

    # Grid search
    params = {'lr__fit_intercept': [True, False], 'pf__degree': [2, 3, 4]}

    lrm_cal_grid = GridSearcher(
        X_train,
        y_train,
        description='linear regression model <<polynomized>> for California housing prices with grid search')
    y_pred_cal_grid = lrm_cal_grid.get_prediction(X_test)
    printAccuracy(
        models=(
            lrm_cal_grid,),
        predictions=(
            y_pred_cal_grid,))

    # reinitialize Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getGoogleShareXy()

    # google shares
    # Linear Regression
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
    printAccuracy(
        models=(
            lrm_gog_intercept,
            lrm_gog_no_intercept),
        predictions=(
            y_pred_lrm_gog_intercept,
            y_pred_lrm_gog_no_intercept))

    # Ridge regression
    rdg_gog_alpha1 = RidgeModel(
        X_train,
        y_train,
        description='ridge regression model with alpha = 1.0 for Google share prices',
        alpha=1.0)
    rdg_gog_alpha10 = RidgeModel(
        X_train,
        y_train,
        description='ridge regression model with alpha = 10.0 for Google share prices',
        alpha=10.0)
    rdg_gog_alpha100 = RidgeModel(
        X_train,
        y_train,
        description='ridge regression model with alpha = 100.0 for Google share prices',
        alpha=100.0)
    y_pred_rdg_gog_alpha1 = rdg_gog_alpha1.get_prediction(X_test)
    y_pred_rdg_gog_alpha10 = rdg_gog_alpha10.get_prediction(X_test)
    y_pred_rdg_gog_alpha100 = rdg_gog_alpha100.get_prediction(X_test)
    printAccuracy(
        models=(
            rdg_gog_alpha1,
            rdg_gog_alpha10,
            rdg_gog_alpha100),
        predictions=(
            y_pred_rdg_gog_alpha1,
            y_pred_rdg_gog_alpha10,
            y_pred_rdg_gog_alpha100))

    # Polynomial stuff + linear regression
    lrm_gog_pol = LinearPipiliner(
        X_train,
        y_train,
        description='linear regression model <<polynomized>> for Google share prices')
    y_pred_gog_pol = lrm_gog_pol.get_prediction(X_test)
    printAccuracy(
        models=(
            lrm_gog_pol,),
        predictions=(
            y_pred_gog_pol,))

    # Grid search
    lrm_gog_grid = GridSearcher(
        X_train,
        y_train,
        description='linear regression model <<polynomized>> for Google share prices with grid search')
    y_pred_gog_grid = lrm_gog_grid.get_prediction(X_test)
    printAccuracy(
        models=(
            lrm_gog_grid,),
        predictions=(
            y_pred_gog_grid,))
