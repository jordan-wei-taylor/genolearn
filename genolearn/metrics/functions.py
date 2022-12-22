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
    return recall(TP, TN, FP, FN) / false_positive_rate(TP, TN, FP, FN)

def negative_likelihood_ratio(TP, TN, FP, FN):
    return false_negative_rate(TP, TN, FP, FN) / specificity(TP, TN, FP, FN)

def prevalence_threshold(TP, TN, FP, FN):
    fpr = false_positive_rate(TP, TN, FP, FN)
    tpr = recall(TP, TN, FP, FN)
    return (fpr ** .5) / ((fpr ** .5) + (tpr ** .5))

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
    return (TP * TN + FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** .5)

def fowlkes_mallows_index(TP, TN, FP, FN):
    return (false_discovery_rate(TP, TN, FP, FN) * recall(TP, TN, FP, FN)) ** .5

def informedness(TP, TN, FP, FN):
    return recall(TP, TN, FP, FN) + specificity(TP, TN, FP, FN) - 1

def markedness(TP, TN, FP, FN):
    return precision(TP, TN, FP, FN) + negative_predictive_value(TP, TN, FP, FN) - 1

def diagnostics_odds_ratio(TP, TN, FP, FN):
    return positive_likelihood_ratio(TP, TN, FP, FN) /  negative_likelihood_ratio(TP, TN, FP, FN)

def count(TP, TN, FP, FN):
    return TP + FN

_metrics = {metric : func for metric, func in locals().items() if not metric.startswith('_')}
