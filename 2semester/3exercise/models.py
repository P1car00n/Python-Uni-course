from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
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


class LGM(Model):

    def __init__(
            self,
            X,
            y,
            description='Logistic regression model',
            **kwargs):
        Model.__init__(self, description,
                       model=LogisticRegression(**kwargs).fit(X, y))


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
                LogisticRegression(
                    solver='lbfgs',
                    multi_class='multinomial',
                    n_jobs=-1),
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
    X_train, X_test, y_train, y_test = data_provider.getMoonsXy()

    # moons
    # Simple logistic regression
    lgm_moon_simple = LGM(
        X_train,
        y_train,
        description='logistic regression model with liblinear solver for the moons dataset',
        solver='liblinear')
    y_pred_lgm_moon_simple = lgm_moon_simple.get_prediction(X_test)
    printAccuracy(
        models=(
            lgm_moon_simple,),
        predictions=(
            y_pred_lgm_moon_simple,))

    # Multinomial logistic regression
    lgm_moon_multi = LGM(
        X_train,
        y_train,
        description='logistic regression model with multinomial solver for the moons dataset',
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1)
    y_pred_lgm_moon_multi = lgm_moon_multi.get_prediction(X_test)
    printAccuracy(
        models=(
            lgm_moon_multi,),
        predictions=(
            y_pred_lgm_moon_multi,))

    lgm_moon_multi_no_penalty = LGM(
        X_train,
        y_train,
        description='no penalty logistic regression model with multinomial solver for the moons dataset',
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        penalty=None)
    y_pred_lgm_moon_multi_no_penalty = lgm_moon_multi_no_penalty.get_prediction(
        X_test)
    printAccuracy(
        models=(
            lgm_moon_multi_no_penalty,),
        predictions=(
            y_pred_lgm_moon_multi_no_penalty,))

    # grid search for multinomial logistic regression
    params = {'max_iter': (100, 200, 500, 1000)}

    lgm_moon_grid = GridSearcher(
        X_train,
        y_train,
        description='grid search logistic regression model with multinomial solver for the moons dataset',
        params=params)
    y_pred_moon_grid = lgm_moon_grid.get_prediction(X_test)
    printAccuracy(
        models=(
            lgm_moon_grid,),
        predictions=(
            y_pred_moon_grid,))

    # reinitialize Xs and ys
    X_train, X_test, y_train, y_test = data_provider.getDigitsXy()

    # digits
    # Simple logistic regression
    lgm_digits_simple = LGM(
        X_train,
        y_train,
        description='logistic regression model with liblinear solver for the digits dataset',
        solver='liblinear')
    y_pred_lgm_digits_simple = lgm_digits_simple.get_prediction(X_test)
    printAccuracy(
        models=(
            lgm_digits_simple,),
        predictions=(
            y_pred_lgm_digits_simple,))

    # Multinomial logistic regression
    lgm_digit_multi = LGM(
        X_train,
        y_train,
        description='logistic regression model with multinomial solver for the digits dataset',
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1)
    y_pred_lgm_digit_multi = lgm_digit_multi.get_prediction(X_test)
    printAccuracy(
        models=(
            lgm_digit_multi,),
        predictions=(
            y_pred_lgm_digit_multi,))

    lgm_digit_multi_no_penalty = LGM(
        X_train,
        y_train,
        description='no penalty logistic regression model with multinomial solver for the digits dataset',
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        penalty=None)
    y_pred_lgm_digit_multi_no_penalty = lgm_digit_multi_no_penalty.get_prediction(
        X_test)
    printAccuracy(
        models=(
            lgm_digit_multi_no_penalty,),
        predictions=(
            y_pred_lgm_digit_multi_no_penalty,))

    # grid search for multinomial logistic regression
    # as above --> params = {'max_iter': (100, 200, 500, 1000)}

    lgm_digit_grid = GridSearcher(
        X_train,
        y_train,
        description='grid search logistic regression model with multinomial solver for the digits dataset',
        params=params)
    y_pred_moon_grid = lgm_digit_grid.get_prediction(X_test)
    printAccuracy(
        models=(
            lgm_digit_grid,),
        predictions=(
            y_pred_moon_grid,))
