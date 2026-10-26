import numpy as np
import pandas as pd
from numba import njit

from .base import Recommender


@njit(cache=True)
def gradients(indptr, items, ratings, day_ids, bins, deviations, mean, parameters, reg):
    p, q, y, drift, daily_p, bu, bi, alpha, daily_b, bin_b = parameters
    gp, gq, gy = np.zeros_like(p), np.zeros_like(q), np.zeros_like(y)
    gd, gdp = np.zeros_like(drift), np.zeros_like(daily_p)
    gbu, gbi, ga = np.zeros_like(bu), np.zeros_like(bi), np.zeros_like(alpha)
    gdb, gbb = np.zeros_like(daily_b), np.zeros_like(bin_b)
    loss = 0.0
    for u in range(p.shape[0]):
        start, end = indptr[u:u + 2]
        norm = 1 / np.sqrt(end - start)
        implicit = np.zeros(p.shape[1])
        implicit_gradient = np.zeros(p.shape[1])
        for row in range(start, end):
            implicit += norm * y[items[row]]
        for row in range(start, end):
            i, day, b, dev = items[row], day_ids[row], bins[row], deviations[row]
            effective = p[u] + drift[u] * dev + daily_p[day] + implicit
            prediction = mean + bu[u] + bi[i] + alpha[u] * dev + daily_b[day] + bin_b[i, b] + np.dot(q[i], effective)
            error = prediction - ratings[row]
            loss += error ** 2
            gp[u] += error * q[i] + reg * p[u]
            gq[i] += error * effective + reg * q[i]
            gd[u] += error * q[i] * dev + reg * drift[u]
            gdp[day] += error * q[i] + reg * daily_p[day]
            gbu[u] += error + reg * bu[u]
            gbi[i] += error + reg * bi[i]
            ga[u] += error * dev + reg * alpha[u]
            gdb[day] += error + reg * daily_b[day]
            gbb[i, b] += error + reg * bin_b[i, b]
            implicit_gradient += error * q[i] * norm
        for row in range(start, end):
            i = items[row]
            gy[i] += implicit_gradient + reg * y[i]
    return (gp, gq, gy, gd, gdp, gbu, gbi, ga, gdb, gbb), np.sqrt(loss / len(items))


class TimeSVDpp(Recommender):
    def __init__(self, factors=16, reg=0.05, lr=0.01, epochs=60, bin_days=90, beta=0.4,
                 temporal=True, implicit=True, seed=42):
        self.factors, self.reg, self.lr, self.epochs = factors, reg, lr, epochs
        self.bin_days, self.beta, self.seed = bin_days, beta, seed
        self.temporal, self.implicit = temporal, implicit

    def fit(self, train, items):
        self._setup(train, items)
        order = np.lexsort((self.i, self.u))
        users, item_ids, ratings = self.u[order], self.i[order], self.r[order]
        days = train.date.to_numpy(dtype='datetime64[D]').astype('int64')[order]
        self.origin = days.min()
        bins = ((days - self.origin) // self.bin_days).astype('int64')
        self.last_bin = bins.max()
        self.indptr = np.r_[0, np.cumsum(np.bincount(users, minlength=self.n_users))]
        self.mean_day = np.bincount(users, weights=days) / np.diff(self.indptr)
        dev = np.sign(days - self.mean_day[users]) * (np.abs(days - self.mean_day[users]) / 365) ** self.beta
        pairs = pd.MultiIndex.from_arrays([users, days])
        day_ids, self.day_pairs = pd.factorize(pairs, sort=True)
        rng = np.random.default_rng(self.seed)
        p = rng.normal(0, 0.03, (self.n_users, self.factors))
        q = rng.normal(0, 0.03, (self.n_items, self.factors))
        y = np.zeros((self.n_items, self.factors))
        drift = np.zeros_like(p)
        daily_p = np.zeros((len(self.day_pairs), self.factors))
        bu, bi, alpha = np.zeros(self.n_users), np.zeros(self.n_items), np.zeros(self.n_users)
        daily_b = np.zeros(len(self.day_pairs))
        bin_b = np.zeros((self.n_items, self.last_bin + 1))
        parameters = (p, q, y, drift, daily_p, bu, bi, alpha, daily_b, bin_b)
        first = [np.zeros_like(x) for x in parameters]
        second = [np.zeros_like(x) for x in parameters]
        self.mean = ratings.mean()
        self.history = []
        for epoch in range(1, self.epochs + 1):
            grads, loss = gradients(self.indptr, item_ids, ratings, day_ids, bins, dev,
                                    self.mean, parameters, self.reg)
            for index, (value, grad) in enumerate(zip(parameters, grads)):
                if (not self.temporal and index in (3, 4, 7, 8, 9)) or (not self.implicit and index == 2):
                    continue
                grad /= len(train)
                first[index] *= 0.9
                first[index] += 0.1 * grad
                second[index] *= 0.999
                second[index] += 0.001 * grad ** 2
                value -= self.lr * (first[index] / (1 - 0.9 ** epoch)) / (np.sqrt(second[index] / (1 - 0.999 ** epoch)) + 1e-8)
            self.history.append(float(loss))
        _, final_loss = gradients(self.indptr, item_ids, ratings, day_ids, bins, dev,
                                  self.mean, parameters, self.reg)
        self.history = self.history[1:] + [float(final_loss)]
        self.p, self.q, self.y, self.drift, self.daily_p, self.bu, self.bi, self.alpha, self.daily_b, self.bin_b = parameters
        matrix = self._matrix(np.ones(len(train)))
        self.implicit_sum = (matrix @ self.y) / np.sqrt(np.diff(matrix.indptr))[:, None]
        return self

    def _score(self, users, items, timestamp):
        day = np.datetime64(timestamp, 'D').astype('int64')
        dev = np.sign(day - self.mean_day[users]) * (np.abs(day - self.mean_day[users]) / 365) ** self.beta
        pairs = pd.MultiIndex.from_arrays([users, np.full(len(users), day)])
        daily_ids = self.day_pairs.get_indexer(pairs)
        daily_b = np.zeros(len(users))
        daily_p = np.zeros((len(users), self.factors))
        known = daily_ids >= 0
        daily_b[known] = self.daily_b[daily_ids[known]]
        daily_p[known] = self.daily_p[daily_ids[known]]
        b = (day - self.origin) // self.bin_days
        item_bias = self.bi[items].copy()
        if 0 <= b <= self.last_bin:
            item_bias += self.bin_b[items, b]
        user_bias = self.bu[users] + self.alpha[users] * dev + daily_b
        effective = self.p[users] + self.drift[users] * dev[:, None] + daily_p + self.implicit_sum[users]
        return self.mean + user_bias[:, None] + item_bias[None, :] + effective @ self.q[items].T
