from   genolearn.logger     import Writing

from   sklearn.ensemble     import AdaBoostClassifier, RandomForestClassifier
from   sklearn.linear_model import LogisticRegression

import numpy as np
import os

class FeatureImportance():
    """
    Feature Importance Class

    Parameters
    ----------
        model : object
            Fitted scikit-learn model.

    Attributes
    ----------

        feature_score : numpy.ndarray
            Numpy array (dtype = float) of feature importance scores.

    Methods
    -------

        ranked_features: numpy.ndarray
            Numpy array (dtype = int) of ranked features.
    """
    def __init__(self, model):
        self.model = model
        self._score, self._rank = _assign(model)
        
    @property
    def feature_scores(self):
        return self._score(self.model)

    def feature_ranks(self, **kwargs):
        return self._rank(self.model, **kwargs)

    def save(self, name, path, **kwargs):
        fullpath = os.path.join(path, name)
        with Writing(fullpath, inline = True):
            np.savez(fullpath, feature_scores = self.feature_scores, feature_ranks = self.feature_ranks(**kwargs))


def _assign(model):
    if isinstance(model, LogisticRegression):
        return _logistic_score, _logistic_rank
    elif isinstance(model, (RandomForestClassifier, AdaBoostClassifier)):
        return _ensemble_score, _ensemble_rank
    else:
        raise Exception()

def _logistic_score(model):
    return model.coef_

def _logistic_rank(model, fun = np.absolute, **kwargs):
    return fun(_logistic_score(model)).mean(axis = -1).argsort()[::-1]

def _ensemble_score(model):
    return model.feature_importances_

def _ensemble_rank(model, **kwargs):
    return _ensemble_score(model).argsort()[::-1]

def feature_importance(meta, feature_selection, model, output):

    from   genolearn.dataloader import DataLoader
    from   genolearn.logger     import Writing, msg

    import pickle

    import pandas as pd
    import numpy  as np

    dataloader = DataLoader(meta)

    selection  = dataloader.load_feature_selection(feature_selection).argsort()

    features   = dataloader.features(selection)

    with open(model, 'rb') as f:
        model = pickle.load(f)

    importance = FeatureImportance(model)

    scores     = importance.feature_scores
    ranks      = importance.feature_ranks()

    features   = np.array(features)[ranks]

    npz        = os.path.join(output, 'importance.npz')
    csv        = os.path.join(output, 'importance-rank.csv')

    os.makedirs(output, exist_ok = True)

    with Writing(npz, inline = True):
        np.savez(npz, features = features, ranks = ranks, scores = scores)

    with Writing(csv, inline = True):
        with open(csv, 'w') as f:
            f.write('\n'.join(features[ranks]))

    msg(f'executed "genolearn feature-importance"')