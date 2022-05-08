import numpy as np

def _base(Y, Y_hat):
    stats  = {}
    unique = set(np.unique(Y)) | set(np.unique(Y_hat))
    for u in sorted(unique):
        TP = sum(Y_hat[...,Y == u] == u)
        TN = sum(Y_hat[...,Y != u] != u)
        FP = sum(Y_hat[...,Y != u] == u)
        FN = sum(Y_hat[...,Y == u] != u)
        stats[u] = TP, TN, FP, FN
    return stats

def recall(TP, TN, FP, FN):
    return TP / (TP + FN)

def specificity(TP, TN, FP, FN):
    return TN / (TN + FP)

def precision(TP, TN, FP, FN):
    return TP / (TP + FP)

def negative_predictive_value(TP, TN, FP, FN):
    return TN / (TN + FN)

def false_negative_rate(TP, TN, FP, FN):
    return FN / (FN + TP)

def false_positive_rate(TP, TN, FP, FN):
    return FP / (FP + TN)

def false_discovery_rate(TP, TN, FP, FN):
    return FP / (FP + TP)

def false_omission_rate(TP, TN, FP, FN):
    return FN / (FN + TN)

def positive_likelihood_ratio(TP, TN, FP, FN):
    fpr = false_positive_rate(TP, TN, FP, FN)
    return np.inf if fpr == 0 else (recall(TP, TN, FP, FN) / fpr)

def negative_likelihood_ratio(TP, TN, FP, FN):
    s = specificity(TP, TN, FP, FN)
    return np.inf if s == 0 else false_negative_rate(TP, TN, FP, FN) / s

def prevalence_threshold(TP, TN, FP, FN):
    fpr = false_positive_rate(TP, TN, FP, FN)
    tpr = recall(TP, TN, FP, FN)
    return np.inf if fpr + tpr == 0 else np.sqrt(fpr) / (np.sqrt(fpr) + np.sqrt(tpr))

def threat_score(TP, TN, FP, FN):
    return TP / (TP + FN + FP)

def prevalence(TP, TN, FP, FN):
    return (TP + FN) / (TP + TN + FP + FN)

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def balanced_accuracy(TP, TN, FP, FN):
    return (recall(TP, TN, FP, FN) + specificity(TP, TN, FP, FN)) / 2

def f1_score(TP, TN, FP, FN):
    return 2 * TP / (2 * TP + FP + FN)

def phi_coefficient(TP, TN, FP, FN):
    return (TP * TN + FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

def fowlkes_mallows_index(TP, TN, FP, FN):
    return np.sqrt(false_discovery_rate(TP, TN, FP, FN) * recall(TP, TN, FP, FN))

def informedness(TP, TN, FP, FN):
    return recall(TP, TN, FP, FN) + specificity(TP, TN, FP, FN) - 1

def markedness(TP, TN, FP, FN):
    return precision(TP, TN, FP, FN) + negative_predictive_value(TP, TN, FP, FN) - 1

def diagnostics_odds_ratio(TP, TN, FP, FN):
    n = negative_likelihood_ratio(TP, TN, FP, FN)
    return np.inf if n == 0 else positive_likelihood_ratio(TP, TN, FP, FN) / n

_metrics = {metric : func for metric, func in locals().items() if not metric.startswith('_') and metric not in ['np']}

def apply(stats, func):
    return {key : func(*value) for key, value in stats.items()}
class Metrics():

    def __init__(self, Y, Y_hat, *metrics):
        stats = _base(Y, Y_hat)
        if len(metrics) == 0:
            self._metric = {metric : apply(stats, func) for metric, func in _metrics.items()}
        else:
            self._metric = {metric : apply(stats, _metrics[metric]) for metric in metrics}

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

def intra_accuracy(Y, Y_hat, labels):
    """
    Returns the intra-class accuracies

    :param Y: target labels
    :type kind: numpy.ndarray of shape (N, M)
    
    :param Y_hat: estimate of labels
    :type kind: numpy.ndarray of shape (N,...,M)

    :param labels: a list of unique labels
    :type kind: list, set, 1-d array

    :return: intra class accuracies.
    :rtype: numpy.ndarray of shape (N,...)
    """
    nan    = np.ones(Y_hat.shape[:-1]) * np.nan
    
    def accuracy(label):
        mask = Y == label
        return (Y_hat[...,mask] == label).mean(axis = -1) if mask.any() else nan

    return np.array(list(map(accuracy, labels)))
