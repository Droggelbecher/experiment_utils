
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import logging

class GenerativeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.estimators_ = []
        self.classes_ = np.unique(y)

        for c in self.classes_:
            grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': np.logspace(-2, 1, 20)})
            grid.fit(X[y == c])
            estimator = grid.best_estimator_
            logging.debug('cls {} bandwidth {}'.format(c, estimator.bandwidth))

            self.estimators_.append(estimator)

    def predict_proba(self, X):
        R = np.zeros((X.shape[0], len(self.estimators_)))
        for j, e in enumerate(self.estimators_):
            R[:, j] = np.exp(e.score_samples(X))

        return R

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis = 1)]


