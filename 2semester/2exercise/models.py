import numpy as np
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
        Model.__init__(self, description, model=LinearRegression(**kwargs).fit(X, y))

    

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# setting thousands since there are numbers like 1,000,111 in the dataset; likewise for dates
dataset_train = pd.read_csv(
    '/home/arthur/Uni/lab2/data/Google_Stock_Price_Train.csv', thousands=',', date_parser='Date')  # input()
dataset_test = pd.read_csv(
    '/home/arthur/Uni/lab2/data/Google_Stock_Price_Test.csv', thousands=',', date_parser='Date')
# here I'm setting what column I want to have predicted
X_dataset_train = dataset_train.drop(['Close', 'Date'], axis=1)
y_dataset_train = dataset_train['Close']
X_dataset_test = dataset_test.drop(['Close', 'Date'], axis=1)
y_dataset_train = dataset_train['Close']
lrm_cal_intercept_google = LinearRegression(
    fit_intercept=True).fit(X_dataset_train, y_dataset_train)
y_pred_lrm_cal_intercept_google = get_prediction(lrm_cal_intercept_google, X_dataset_test)
print("Prediction accuracy (R^2) with intercept; google: ",
      lrm_cal_intercept_google.score(X_dataset_train, y_dataset_train))
print("Prediction accuracy (R^2) with intercept; google y_test: ",
      lrm_cal_intercept_google.score(X_dataset_train, y_dataset_train))


lrm_cal_intercept = LinearRegression(fit_intercept=True).fit(X_train, y_train)
lrm_cal_no_intercept = LinearRegression(
    fit_intercept=False).fit(
        X_train, y_train)

y_pred_lrm_cal_intercept = get_prediction(lrm_cal_intercept, X_test)
# Old: y_pred_lrm_cal_no_intercapt = lrm_cal_no_intercept.predict(X_test)
y_pred_lrm_cal_no_intercept = get_prediction(lrm_cal_no_intercept, X_test)

print("Prediction accuracy (R^2) with intercept: ",
      lrm_cal_intercept.score(X_train, y_train))
print("Prediction accuracy (R^2) with no intercept: ",
      lrm_cal_no_intercept.score(X_train, y_train))
print("Prediction accuracy (R^2) with intercept for y_test: ",
      r2_score(y_test, y_pred_lrm_cal_intercept))
print("Prediction accuracy (R^2) with no intercept for y_test: ",
      r2_score(y_test, y_pred_lrm_cal_no_intercept))
# print(lrm_cal_intercept.score(y_test, lrm_cal_intercept.predict(X_test)))
# print(lrm_cal_no_intercept.score(y_test, lrm_cal_no_intercept.predict(X_test)))

if __name__ == '__main__':
    pass