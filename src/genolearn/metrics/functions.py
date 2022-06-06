import numpy as np

def recall(TP, TN, FP, FN):
    r"""
    recall metric
    
        :math:`\frac{TP}{TP + FN}`
    """
    return TP / (TP + FN)

def specificity(TP, TN, FP, FN):
    r"""
    specificity metric
    
        :math:`\frac{TN}{TN + FP}`
    """
    return TN / (TN + FP)

def precision(TP, TN, FP, FN):
    r"""
    precision metric
    
        :math:`\frac{TP}{TP + FP}`
    """
    return TP / (TP + FP)

def negative_predictive_value(TP, TN, FP, FN):
    r"""
    negative predictive value
    
        :math:`\frac{TN}{TN + FN}`
    """
    return TN / (TN + FN)

def false_negative_rate(TP, TN, FP, FN):
    r"""
    false negative rate
    
        :math:`\frac{FN}{FN + TP}`
    """
    return FN / (FN + TP)

def false_positive_rate(TP, TN, FP, FN):
    r"""
    false positive rate
    
        :math:`\frac{FP}{FP + TN}`
    """
    return FP / (FP + TN)

def false_discovery_rate(TP, TN, FP, FN):
    r"""
    false discovery rate
    
        :math:`\frac{FP}{FP + TP}`
    """
    return FP / (FP + TP)

def false_omission_rate(TP, TN, FP, FN):
    r"""
    false omission rate
    
        :math:`\frac{FN}{FN + TN}`
    """
    return FN / (FN + TN)

def positive_likelihood_ratio(TP, TN, FP, FN):
    r"""
    positive likelihood ratio
    
        :math:`\frac{TP}{TP + FN} \frac{FP + TN}{FP}`
    """
    return recall(TP, TN, FP, FN) / false_positive_rate(TP, TN, FP, FN)

def negative_likelihood_ratio(TP, TN, FP, FN):
    r"""
    negative likelihood ratio
    
        :math:`\frac{FN}{FN + TP} \frac{TN + FP}{TN}`
    """
    return false_negative_rate(TP, TN, FP, FN) / specificity(TP, TN, FP, FN)

def prevalence_threshold(TP, TN, FP, FN):
    r"""
    prevalence threshold
    
        :math:`\frac{\sqrt{\frac{FP}{FP + TN}}}{\sqrt{\frac{FP}{FP + TN}} + \sqrt{\frac{TP}{TP + FN}}}`
    """
    fpr = false_positive_rate(TP, TN, FP, FN)
    tpr = recall(TP, TN, FP, FN)
    return np.sqrt(fpr) / (np.sqrt(fpr) + np.sqrt(tpr))

def threat_score(TP, TN, FP, FN):
    r"""
    threat score
    
        :math:`\frac{TP}{TP + FN + FP}`
    """
    return TP / (TP + FN + FP)

def prevalence(TP, TN, FP, FN):
    r"""
    prevalence metric
    
        :math:`\frac{TP + FN}{TP + TN + FP + FN}`
    """
    return (TP + FN) / (TP + TN + FP + FN)

def accuracy(TP, TN, FP, FN):
    r"""
    accuracy metric
    
        :math:`\frac{TP + TN}{TP + TN + FP + FN}`
    """
    return (TP + TN) / (TP + TN + FP + FN)

def balanced_accuracy(TP, TN, FP, FN):
    r"""
    balanced accuracy
    
        :math:`\frac{1}{2} \left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)`
    """
    return (recall(TP, TN, FP, FN) + specificity(TP, TN, FP, FN)) / 2

def f1_score(TP, TN, FP, FN):
    r"""
    f1 score
    
        :math:`\frac{2TP}{2TP + FP + FN}`
    """
    return 2 * TP / (2 * TP + FP + FN)

def phi_coefficient(TP, TN, FP, FN):
    r"""
    phi coefficient
    
        :math:`\frac{TP \times TN + FP \times FN}{\sqrt{(TP + FP) (TP + FN) (TN + FP) (TN + FN)}}`
    """
    return (TP * TN + FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

def fowlkes_mallows_index(TP, TN, FP, FN):
    r"""
    fowlkes mallows index

        :math:`\sqrt{\frac{FP}{FP + TP} \frac{TP}{TP + FN}}`
    """
    return np.sqrt(false_discovery_rate(TP, TN, FP, FN) * recall(TP, TN, FP, FN))

def informedness(TP, TN, FP, FN):
    r"""
    informedness metric
    
        :math:`\frac{TP}{TP + FN} + \frac{TN}{TN + FP} - 1`
    """
    return recall(TP, TN, FP, FN) + specificity(TP, TN, FP, FN) - 1

def markedness(TP, TN, FP, FN):
    r"""
    markedness metric
    
        :math:`\frac{TP}{TP + FP} + \frac{TN}{TN + FN}`
    """
    return precision(TP, TN, FP, FN) + negative_predictive_value(TP, TN, FP, FN) - 1

def diagnostics_odds_ratio(TP, TN, FP, FN):
    r"""
    diagnostics odds ratio
    
        :math:`\frac{TP}{TP + FN} \frac{FP + TN}{FP} \frac{FN + TP}{FN} \frac{TN}{TN + FP}`
    """
    return positive_likelihood_ratio(TP, TN, FP, FN) /  negative_likelihood_ratio(TP, TN, FP, FN)

def count(TP, TN, FP, FN):
    r"""
    count metric
    
        :math:`TP + TN`
    """
    return TP + FN

_metrics = {metric : func for metric, func in locals().items() if not metric.startswith('_') and metric not in ['np']}
