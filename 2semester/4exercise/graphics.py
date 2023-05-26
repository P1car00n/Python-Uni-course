import matplotlib.pyplot as plt
from sklearn import svm, naive_bayes
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_blobs, make_circles, make_moons, fetch_covtype


def getBlobsXy(test_size=0.2):
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [1.5, 0.5]
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers,
                      cluster_std=clusters_std, random_state=0, shuffle=False)
    return X, y


def getCirclesXy(test_size=0.2):
    X, y = make_circles(500, factor=.1, noise=.1)
    return X, y


def getMoonsXy(test_size=0.2):
    X, y = make_moons(500, noise=.1)
    return X, y


def getCovtypesXy(test_size=0.2):
    # TODO: replace with input()
    X, y = fetch_covtype(
        data_home='/home/arthur/Developing/Python-Uni-course/2semester/4exercise/data', return_X_y=True)
    return X, y


X, y = getBlobsXy()
#circlesX, ciclesY = getCirclesXy()
#moonsX, moonsY = getMoonsXy()
#covtypesX, covtypesY = getCovtypesXy()

models = (
    naive_bayes.GaussianNB(),
    svm.SVC(kernel="linear", C=1.0),
    svm.SVC(kernel="linear", C=5.0),
    svm.LinearSVC(C=1.0, max_iter=10000),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=1.0),
    svm.SVC(kernel="poly", degree=6, gamma="auto", C=1.0)
)
models = (clf.fit(X, y) for clf in models)

titles = (
    "Naive Bayes",
    "C-Support Linear Vector Classification; C=1.0",
    "C-Support Linear Vector Classification; C=5.0",
    "Linear Support Vector Classification",
    "C-Support Multinomial Vector Classification; degree=3",
    "C-Support Multinomial Vector Classification; degree=6",
    "C-Support Multinomial Vector Classification; degree=6, C=4.0",
)

fig, sub = plt.subplots(2, 3)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
