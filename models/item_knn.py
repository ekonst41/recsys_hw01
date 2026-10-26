import numpy as np
from scipy.sparse import csr_matrix

from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=100, shrink=20):
        self.neighbors = neighbors
        self.shrink = shrink

    def fit(self, train, items):
        self._setup(train, items)
        self.x = self._matrix((self.r >= 4).astype(float))
        self.x.eliminate_zeros()
        counts = (self.x.T @ self.x).tocsr()
        norm = np.sqrt(self.x.power(2).sum(axis=0)).A1
        rows, cols, values = [], [], []
        for i in range(self.n_items):
            start, end = counts.indptr[i:i + 2]
            ids, common = counts.indices[start:end], counts.data[start:end]
            scores = common / np.maximum(norm[i] * norm[ids], 1e-12)
            scores *= common / (common + self.shrink)
            scores[ids == i] = 0
            order = np.lexsort((ids, -scores))[:self.neighbors]
            order = order[scores[order] > 0]
            rows.extend([i] * len(order))
            cols.extend(ids[order])
            values.extend(scores[order])
        self.similarity = csr_matrix((values, (rows, cols)), shape=(self.n_items, self.n_items))
        return self

    def _score(self, users, items, timestamp):
        return (self.x[users] @ self.similarity[items].T).toarray()
