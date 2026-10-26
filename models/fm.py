import ast
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from .base import Recommender


def parse_list(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (SyntaxError, ValueError):
        return []

def fm_epoch(order, users, items, ratings, indptr, indices, values, offset, w, v, bias, lr, reg):
    loss = 0.0
    dimensions = v.shape[1]
    for row in order:
        user, item = users[row], items[row]
        start, end = indptr[item], indptr[item + 1]
        total = v[user].copy()
        square = v[user] ** 2
        score = bias + w[user]
        for pos in range(start, end):
            feature, value = offset + indices[pos], values[pos]
            score += w[feature] * value
            for f in range(dimensions):
                total[f] += v[feature, f] * value
                square[f] += (v[feature, f] * value) ** 2
        score += 0.5 * np.sum(total ** 2 - square)
        error = ratings[row] - score
        loss += error ** 2
        bias += lr * error
        w[user] += lr * (error - reg * w[user])
        for f in range(dimensions):
            v[user, f] += lr * (error * (total[f] - v[user, f]) - reg * v[user, f])
        for pos in range(start, end):
            feature, value = offset + indices[pos], values[pos]
            w[feature] += lr * (error * value - reg * w[feature])
            for f in range(dimensions):
                v[feature, f] += lr * (error * value * (total[f] - v[feature, f] * value) - reg * v[feature, f])
    return bias, np.sqrt(loss / len(order))


class FM(Recommender):
    def __init__(self, factors=16, reg=0.02, lr=0.005, epochs=20, max_tags=256, metadata=True, seed=42):
        self.factors, self.reg, self.lr, self.epochs = factors, reg, lr, epochs
        self.max_tags, self.use_metadata, self.seed = max_tags, metadata, seed

    def _features(self):
        tags = self.metadata.tags.map(parse_list)
        counts = Counter(tag for row in tags for tag in set(row))
        vocabulary = sorted(counts, key=lambda tag: (-counts[tag], tag))[:self.max_tags]
        self.tag_vocabulary = {tag: i for i, tag in enumerate(vocabulary)}
        rows, cols, values = [], [], []
        for item, (_, row) in enumerate(self.metadata.iterrows()):
            features = [(item, 1.0)]
            if self.use_metadata:
                selected = sorted({self.tag_vocabulary[tag] for tag in tags.iloc[item] if tag in self.tag_vocabulary})
                features += [(self.n_items + tag, 1 / np.sqrt(max(len(selected), 1))) for tag in selected]
                for field, boundaries in [('minutes', [0, 15, 30, 60, 120, 1440]),
                                          ('n_ingredients', [0, 5, 10, 15, 20, 30])]:
                    value = row[field]
                    bucket = 7 if not np.isfinite(value) or value < 0 or (field == 'minutes' and value > 10080) else np.searchsorted(boundaries, value, side='right')
                    field_offset = 0 if field == 'minutes' else 8
                    features.append((self.n_items + len(vocabulary) + field_offset + bucket, 1.0))
            rows.extend([item] * len(features))
            cols.extend(feature for feature, value in features)
            values.extend(value for feature, value in features)
        size = self.n_items + len(vocabulary) + 16 if self.use_metadata else self.n_items
        return csr_matrix((values, (rows, cols)), shape=(self.n_items, size))

    def fit(self, train, items):
        self._setup(train, items)
        self.features = self._features()
        rng = np.random.default_rng(self.seed)
        size = self.n_users + self.features.shape[1]
        self.w = np.zeros(size)
        self.v = rng.normal(0, 0.03, (size, self.factors))
        self.bias = self.r.mean()
        self.history = []
        for epoch in range(self.epochs):
            self.bias, _ = fm_epoch(rng.permutation(len(train)), self.u, self.i, self.r,
                                      self.features.indptr, self.features.indices, self.features.data,
                                      self.n_users, self.w, self.v, self.bias,
                                      self.lr / (1 + 0.03 * epoch), self.reg)
            matrix = self.features
            item_v = self.v[self.n_users:]
            self.item_sum = matrix @ item_v
            self.item_bias = self.bias + matrix @ self.w[self.n_users:]
            self.item_bias += 0.5 * (self.item_sum ** 2 - matrix.power(2) @ (item_v ** 2)).sum(axis=1)
            prediction = self.w[self.u] + self.item_bias[self.i] + (self.v[self.u] * self.item_sum[self.i]).sum(axis=1)
            self.history.append(float(np.sqrt(np.mean((self.r - prediction) ** 2))))
        return self

    def _score(self, users, items, timestamp):
        return self.w[users, None] + self.item_bias[None, items] + self.v[users] @ self.item_sum[items].T
