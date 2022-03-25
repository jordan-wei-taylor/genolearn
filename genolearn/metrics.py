import numpy as np

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
