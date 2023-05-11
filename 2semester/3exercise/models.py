from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LogisticRegression

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
