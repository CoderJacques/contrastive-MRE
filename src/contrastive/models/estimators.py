import numpy as np
import multiprocessing
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


class AgeEstimator(BaseEstimator):
    """ Define the age estimator on latent space network features.
    """

    def __init__(self):
        n_jobs = multiprocessing.cpu_count()
        self.age_estimator = GridSearchCV(
            Ridge(), param_grid={"alpha": 10. ** np.arange(-2, 3)}, cv=5,
            scoring="r2", n_jobs=n_jobs)

    def fit(self, x, y):
        self.age_estimator.fit(x, y)
        return self.score(x, y)

    def predict(self, x):
        y_pred = self.age_estimator.predict(x)
        return y_pred

    def score(self, x, y):
        y_pred = self.age_estimator.predict(x)
        return mean_absolute_error(y, y_pred)