from   genolearn.metrics.functions import _metrics
import numpy as np

def _base(Y, Y_hat):
    """ base function to compute statistics required for metric functions in .funcs.py """
    stats  = {}
    unique = set(np.unique(Y)) | set(np.unique(Y_hat))
    for u in sorted(unique):
        TP = sum(Y_hat[...,Y == u] == u)
        TN = sum(Y_hat[...,Y != u] != u)
        FP = sum(Y_hat[...,Y != u] == u)
        FN = sum(Y_hat[...,Y == u] != u)
        stats[u] = TP, TN, FP, FN
    return stats


def _apply(stats, func):
    """ applies func with stats dictionary from _base """
    return {key : func(*value) for key, value in stats.items()}

def _func(name):

    if isinstance(name, function):
        return name

    elif name is None:
        return name

    def weighted_mean(score, w):
        return (score * w) / w.sum()

    funcs = dict(mean = np.mean, weighted_mean = weighted_mean)

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
        stats = _base(Y, Y_hat)
        if len(metrics) == 0:
            self._metric = {metric : _apply(stats, func) for metric, func in _metrics.items()}
        else:
            self._metric = {metric : _apply(stats, _metrics[metric]) for metric in metrics}
        self._weight = np.unique(Y, return_counts = True)[1]

    def __call__(self, *keys, func = None):
        func = _func(func)
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
