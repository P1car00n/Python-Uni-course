import pandas as pd
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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


X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# should put it in a separate module
# setting thousands since there are numbers like 1,000,111 in the dataset;
# likewise for dates
dataset_train = pd.read_csv(
    '/home/arthur/Uni/lab2/data/Google_Stock_Price_Train.csv',
    thousands=',',
    date_format='%m/%d/%Y')  # input()
dataset_test = pd.read_csv(
    '/home/arthur/Uni/lab2/data/Google_Stock_Price_Test.csv',
    thousands=',',
    date_format='%m/%d/%Y')
# here I'm setting what column I want to have predicted
X_dataset_train = dataset_train.drop(['Close', 'Date'], axis=1)
y_dataset_train = dataset_train['Close']
X_dataset_test = dataset_test.drop(['Close', 'Date'], axis=1)
y_dataset_test = dataset_test['Close']


if __name__ == '__main__':
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

    # google shares
    lrm_gog_intercept = LRM(
        X_dataset_train,
        y_dataset_train,
        description='linear regression model with intercept for Google share prices',
        fit_intercept=True)
    lrm_gog_no_intercept = LRM(
        X_dataset_train,
        y_dataset_train,
        description='linear regression model with no intercept for Google share prices',
        fit_intercept=False)
    y_pred_lrm_gog_intercept = lrm_gog_intercept.get_prediction(X_dataset_test)
    y_pred_lrm_gog_no_intercept = lrm_gog_no_intercept.get_prediction(
        X_dataset_test)
    print("Prediction accuracy (R^2) for", lrm_gog_intercept,
          'is', lrm_gog_intercept.get_score(X_dataset_train, y_dataset_train))
    print(
        "Prediction accuracy (R^2) for",
        lrm_gog_no_intercept,
        'is',
        lrm_gog_no_intercept.get_score(
            X_dataset_train,
            y_dataset_train))
    print(
        "Prediction accuracy (R^2) of y_test for",
        lrm_gog_intercept,
        'is',
        r2_score(
            y_dataset_test,
            y_pred_lrm_gog_intercept))
    print(
        "Prediction accuracy (R^2) of y_test for",
        lrm_gog_intercept,
        'is',
        r2_score(
            y_dataset_test,
            y_pred_lrm_gog_no_intercept))
