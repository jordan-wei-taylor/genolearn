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

    def __call__(self, *keys, func = None):
        if keys:
            if func:
                return {key : func(np.array(list(self._metric[key].values()))) for key in keys}
            return {key : self._metric[key] for key in keys}
        elif func:
            return {key : func(np.array(list(value.values()))) for key, value in self._metric.items()}
        return self._metric

    def __repr__(self):
        if len(self._metric) == 1:
            Key = list(self._metric)[0]
            sub = '{' + ', '.join([f'{key} : {value:.3f}' for key, value in self._metric[Key].items()]) + '}'
            return f'Metrics({Key} : {sub})'
        keys = list(self._metric)
        return f'Metrics({", ".join(keys[:3])}, ...)' if len(keys) > 3 else f'Metrics({", ".join(keys[:3])})'

    def __getitem__(self, name):
        return self._metric[name]
