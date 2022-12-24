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
        self._detailed_scores, self._aggregate_scores = _assign(model)
        
    @property
    def detailed_scores(self):
        return self._detailed_scores(self.model)

    @property
    def aggregate_scores(self):
        return self._aggregate_scores(self.model)  

def _assign(model):
    if isinstance(model, LogisticRegression):
        return _logistic_detailed_scores, _logistic_aggregate_scores
    elif isinstance(model, (RandomForestClassifier, AdaBoostClassifier)):
        return _ensemble_detailed_scores, _ensemble_aggregate_scores
    else:
        raise Exception()

def _logistic_detailed_scores(model):
    return model.coef_

def _logistic_aggregate_scores(model, fun = np.absolute, **kwargs):
    return fun(_logistic_detailed_scores(model)).mean(axis = 0)

def _ensemble_detailed_scores(model):
    return model.feature_importances_

def _ensemble_aggregate_scores(model, **kwargs):
    return _ensemble_detailed_scores(model)

def feature_importance(meta, feature_selection, num_features, output):

    from   genolearn.dataloader import DataLoader
    from   genolearn.logger     import Writing, msg

    import pickle

    import pandas as pd
    import numpy  as np

    dataloader = DataLoader(meta)

    selection  = dataloader.load_feature_selection(feature_selection).argsort()

    features   = dataloader.features(selection[:num_features])

    for model, pre in zip(['model.pickle', 'full-model.pickle'], ['', 'full-']):
        with open(os.path.join(os.path.dirname(output), model), 'rb') as f:
            model = pickle.load(f)

        importance = FeatureImportance(model)

        d_scores   = importance.detailed_scores
        a_scores   = importance.aggregate_scores

        ranks      = a_scores.argsort()[::-1]

        features   = np.array(features)

        npz        = os.path.join(output, f'{pre}importance.npz')
        csv        = os.path.join(output, f'{pre}importance-rank.csv')

        os.makedirs(output, exist_ok = True)

        with Writing(npz, inline = True):
            np.savez(npz, features = features, ranks = ranks, detailed_scores = d_scores, aggregate_scores = a_scores)

        with Writing(csv, inline = True):
            df = pd.DataFrame()
            df['features'] = features[ranks]
            df['aggregate_score'] = a_scores[ranks]
            df.to_csv(csv, index  = False)

    msg(f'executed "genolearn feature-importance"')