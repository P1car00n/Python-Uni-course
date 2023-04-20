import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# potentially add some oop
def get_prediction(model, samples):
    return model.predict(samples)


# now could use it for the google thing, since it's in csv # for
# predicting may use my fnd where I split the data into training and
# predicting
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lrm_cal_intercept = LinearRegression(fit_intercept=True).fit(X_train, y_train)
lrm_cal_no_intercept = LinearRegression(
    fit_intercept=False).fit(
        X_train, y_train)

y_pred_lrm_cal_intercapt = get_prediction(lrm_cal_intercept, X_test)
# Old: y_pred_lrm_cal_no_intercapt = lrm_cal_no_intercept.predict(X_test)
y_pred_lrm_cal_no_intercapt = get_prediction(lrm_cal_no_intercept, X_test)

print("Prediction accuracy (R^2) with intercept: ", lrm_cal_intercept.score(X_train, y_train))
print("Prediction accuracy (R^2) with no intercept: ", lrm_cal_no_intercept.score(X_train, y_train))
print("Prediction accuracy (R^2) with intercept for y_test: ", r2_score(y_test, y_pred_lrm_cal_intercapt))
print("Prediction accuracy (R^2) with no intercept for y_test: ", r2_score(y_test, y_pred_lrm_cal_no_intercapt))
# print(lrm_cal_intercept.score(y_test, lrm_cal_intercept.predict(X_test)))
# print(lrm_cal_no_intercept.score(y_test, lrm_cal_no_intercept.predict(X_test)))
