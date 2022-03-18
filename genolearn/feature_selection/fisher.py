from   genolearn.feature_selection.base import _feature_selection
from   genolearn.logger                 import msg

import numpy as np


def fisher_score(dataloader, values):

    def init(dataloader):
        encode = {c : i for i, c in enumerate(set(dataloader.meta[dataloader.target]))}

        n = np.zeros(dataloader.c)
        m = np.zeros(dataloader.m)
        M = np.zeros((dataloader.c, dataloader.m))
        v = np.zeros((dataloader.c, dataloader.m))

        return {}, (encode, n, m, M, v), {}

    def inner_loop(ret, i, x, label, value, encode, n, m, M, v, **kwargs):
        msg(f'{value} : {i}', inline = True)
        y = encode[label]

        n[y] += 1
        m    += x
        M[y] += x
        v[y] += np.square(x)

    def outer_loop(ret, i, value, encode, n, m, M, v, **kwargs):

        msg(f'{value} : {i} (computing fisher)', inline = True)

        n   = n.reshape(-1, 1)
        m   = m / dataloader.n
        M   = np.divide(M, n, where = n > 0)
        v   = np.divide(v, n, where = n > 0)
        v   = v - np.square(M)

        n   = n.reshape(-1)
        num = n @ np.square(M - m)
        den = n @ v

        S   = np.divide(num, den, where = (den > 0))

        msg(f'{value} : {i} (completed)')

        ret[value] = S

    return _feature_selection(dataloader, init, inner_loop, outer_loop, values, force_dense = True)