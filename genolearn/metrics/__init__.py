from   genolearn.metrics.functions import _metrics
import numpy as np

def _base(Y, Y_hat):
    """ base function to compute statistics required for metric functions in .funcs.py """
    stats  = {}
    unique = set(np.unique(Y)) | set(np.unique(Y_hat))
    for u in sorted(unique):
        TP = (Y_hat[...,Y == u] == u).sum(axis = -1)
        TN = (Y_hat[...,Y != u] != u).sum(axis = -1)
        FP = (Y_hat[...,Y != u] == u).sum(axis = -1)
        FN = (Y_hat[...,Y == u] != u).sum(axis = -1)
        stats[u] = TP, TN, FP, FN
    return stats

def _apply(stats, func):
    """ applies func with stats dictionary from _base """
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        return {key : func(*value) for key, value in stats.items()}

def _func(name, count):

    def weighted_mean(score):
        return np.einsum('n...,n', score, count) / count.sum()

    funcs = {'mean' : np.mean, 'weighted_mean' : weighted_mean}

    if callable(name) or name is None:
        return name

    if name in funcs:
        return funcs[name]

    raise Exception()

class Metrics():
    """
    Metrics Class
    
    Computes desired metrics given target labels and predictions.

    Parameters
    ----------
        Y : str
            Array with shape (N,) containing target values.

        Y_hat : str
            Array with shape (...,N) containing predicted values.

        *metrics : str
            String of metrics to use. If None, all metrics are computed.
    """
    def __init__(self, Y, Y_hat, *metrics):
        Y, Y_hat = map(np.array, [Y, Y_hat])
        stats    = _base(Y, Y_hat)
        if len(metrics) == 0:
            self._metric = {metric : _apply(stats, func) for metric, func in _metrics.items()}
        else:
            metrics      = set(metrics) | {'count'}
            self._metric = {metric : _apply(stats, _metrics[metric]) for metric in metrics}
        self.count = self._metric.pop('count')
        
    @property
    def _count(self):
        return np.array(list(self.count.values()))

    def __call__(self, *keys, func = None):
        func = _func(func, self._count)
        ret  = None
        if keys:
            if func:
                ret = {key : func(np.array(list(self._metric[key].values()))) for key in keys}
            else:
                ret = {key : self._metric[key] for key in keys}
        elif func:
            ret = {key : func(np.array(list(value.values()))) for key, value in self._metric.items()}
        else:
            ret = self._metric

        if ret:
            if len(ret) == 1:
                return ret[list(ret)[0]]

        return ret

    def __repr__(self):
        if len(self._metric) == 1:
            Key = list(self._metric)[0]
            sub = '{' + ', '.join([f'{key} : {value:.3f}' for key, value in self._metric[Key].items()]) + '}'
            return f'Metrics({Key} : {sub})'
        keys = list(self._metric)
        return f'Metrics({", ".join(keys[:3])}, ...)' if len(keys) > 3 else f'Metrics({", ".join(keys[:3])})'

    def __getitem__(self, name):
        return self._metric[name]
