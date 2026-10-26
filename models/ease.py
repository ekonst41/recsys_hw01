import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .base import Recommender


class EASE(Recommender):
    def __init__(self, reg=100):
        self.reg = reg

    def fit(self, train, items):
        self._setup(train, items)
        self.x = self._matrix((self.r >= 4).astype(float))
        gram = (self.x.T @ self.x).toarray()
        gram.flat[::self.n_items + 1] += self.reg
        precision = cho_solve(cho_factor(gram, check_finite=False), np.eye(self.n_items), check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def _score(self, users, items, timestamp):
        return self.x[users] @ self.weights[:, items]
