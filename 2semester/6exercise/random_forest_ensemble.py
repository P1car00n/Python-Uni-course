from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import data_provider

X_train, X_test, y_train, y_test = data_provider.getBlobsXy(return_no_control=True)

# Create a random forest classifier with different parameter values
rf1 = RandomForestClassifier(
    max_depth=3,
    max_features=4,
    bootstrap=True,
    n_estimators=50)
rf2 = RandomForestClassifier(
    max_depth=5,
    max_features=6,
    bootstrap=True,
    n_estimators=100)
rf3 = RandomForestClassifier(
    max_depth=10,
    max_features=8,
    bootstrap=True,
    n_estimators=200)
rf4 = RandomForestClassifier(
    max_depth=10,
    max_features=8,
    bootstrap=False,
    n_estimators=200)

# Create a voting classifier with the random forest classifiers
voting_clf_hard = VotingClassifier(
    estimators=[
        ('rf1', rf1), ('rf2', rf2), ('rf3', rf3), ('rf4', rf4)], voting='hard')

voting_clf_soft = VotingClassifier(
    estimators=[
        ('rf1', rf1), ('rf2', rf2), ('rf3', rf3), ('rf4', rf4)], voting='soft')

# Fit the voting classifier on the training data
voting_clf_hard.fit(X_train, y_train)
voting_clf_soft.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_hard = voting_clf_hard.predict(X_test)
y_pred_soft = voting_clf_soft.predict(X_test)

# Evaluate the accuracy of the ensemble
accuracy = [accuracy_score(y_test, y_pred_hard), accuracy_score(y_test, y_pred_soft)]
print("Accuracy for hard and soft voting:", accuracy)
