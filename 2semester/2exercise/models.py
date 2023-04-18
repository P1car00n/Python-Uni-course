import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


X, y = fetch_california_housing(return_X_y=True) # now could use it for the google thing, since it's in csv # for predicting may use my fnd where I split the data into training and predicting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


lrm_cal_intercept = LinearRegression(fit_intercept=True).fit(X_train, y_train)
lrm_cal_no_intercept = LinearRegression(fit_intercept=False).fit(X_train, y_train)

y_pred = lrm_cal_intercept.predict(X_test)

print(lrm_cal_intercept.score(X_train, y_train))
print(lrm_cal_no_intercept.score(X_train, y_train))
print(r2_score(y_test, y_pred))
#print(lrm_cal_intercept.score(y_test, lrm_cal_intercept.predict(X_test)))
#print(lrm_cal_no_intercept.score(y_test, lrm_cal_no_intercept.predict(X_test)))