import numpy as np
from scipy.linalg import solve

from .base import Recommender


class ALS(Recommender):
    def __init__(self, factors=32, reg=10, epochs=15, seed=42):
        self.factors, self.reg, self.epochs, self.seed = factors, reg, epochs, seed

    def _update(self, matrix, other, other_bias):
        factors = np.zeros((matrix.shape[0], self.factors))
        bias = np.zeros(matrix.shape[0])
        penalty = self.reg * np.eye(self.factors + 1)
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row:row + 2]
            ids = matrix.indices[start:end]
            design = np.column_stack([other[ids], np.ones(len(ids))])
            target = matrix.data[start:end] - self.mean - other_bias[ids]
            solution = solve(design.T @ design + penalty, design.T @ target, assume_a='pos')
            factors[row], bias[row] = solution[:-1], solution[-1]
        return factors, bias

    def fit(self, train, items):
        self._setup(train, items)
        rng = np.random.default_rng(self.seed)
        self.mean = self.r.mean()
        self.p = rng.normal(0, 0.1, (self.n_users, self.factors))
        self.q = rng.normal(0, 0.1, (self.n_items, self.factors))
        self.bu, self.bi = np.zeros(self.n_users), np.zeros(self.n_items)
        matrix = self._matrix(self.r)
        transposed = matrix.T.tocsr()
        self.history = []
        self.objective_history = []
        for _ in range(self.epochs):
            self.p, self.bu = self._update(matrix, self.q, self.bi)
            self.q, self.bi = self._update(transposed, self.p, self.bu)
            prediction = self.mean + self.bu[self.u] + self.bi[self.i] + (self.p[self.u] * self.q[self.i]).sum(axis=1)
            self.history.append(float(np.sqrt(np.mean((self.r - prediction) ** 2))))
            penalty = sum(np.sum(value ** 2) for value in (self.p, self.q, self.bu, self.bi))
            self.objective_history.append(float(np.sum((self.r - prediction) ** 2) + self.reg * penalty))
        return self

    def _score(self, users, items, timestamp):
        return self.mean + self.bu[users, None] + self.bi[None, items] + self.p[users] @ self.q[items].T
