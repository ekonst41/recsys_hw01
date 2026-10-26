import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class Recommender:
    def _setup(self, train, items):
        self.user_ids = np.sort(train.user_id.unique())
        self.item_ids = np.sort(train.recipe_id.unique())
        self.users = pd.Index(self.user_ids)
        self.items = pd.Index(self.item_ids)
        self.n_users, self.n_items = len(self.users), len(self.items)
        self.train = train.copy()
        self.u = self.users.get_indexer(train.user_id)
        self.i = self.items.get_indexer(train.recipe_id)
        self.r = train.rating.to_numpy(dtype=np.float64)
        self.timestamp = train.date.max() + pd.Timedelta(days=1)
        self.metadata = items.set_index('id').reindex(self.item_ids)
        return self

    def _matrix(self, values):
        return csr_matrix((values, (self.u, self.i)), shape=(self.n_users, self.n_items))

    def predict(self, user_ids, item_ids=None, timestamp=None):
        users = self.users.get_indexer(np.atleast_1d(user_ids))
        items = np.arange(self.n_items) if item_ids is None else self.items.get_indexer(np.atleast_1d(item_ids))
        if (users < 0).any() or (items < 0).any():
            raise ValueError('Неизвестный user_id или recipe_id')
        time = self.timestamp if timestamp is None else pd.Timestamp(timestamp)
        return np.asarray(self._score(users, items, time), dtype=np.float64)

    def recommend(self, user_ids, k=10, seen=None, timestamp=None):
        user_ids = np.atleast_1d(user_ids)
        scores = self.predict(user_ids, timestamp=timestamp)
        history = self.train if seen is None else seen
        row = pd.Index(user_ids).get_indexer(history.user_id)
        col = self.items.get_indexer(history.recipe_id)
        valid = (row >= 0) & (col >= 0)
        scores[row[valid], col[valid]] = -np.inf
        order = np.argsort(-scores, axis=1, kind='stable')[:, :k]
        result = self.item_ids[order].copy()
        result[~np.isfinite(np.take_along_axis(scores, order, axis=1))] = -1
        return result
