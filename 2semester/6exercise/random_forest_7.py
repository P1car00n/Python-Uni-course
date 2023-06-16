from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import data_provider

X_train, X_test, y_train, y_test = data_provider.getBlobsXy(
    return_no_control=True)

rf = RandomForestClassifier()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params, best_model)
