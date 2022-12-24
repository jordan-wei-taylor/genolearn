from   genolearn.models  import classification
from   genolearn.logger  import Waiting, msg
from   genolearn.utils   import monitor_RAM
from   genolearn.metrics import Metrics
from   itertools         import product
from   time              import time

import numpy as np
import joblib
import os

root = 'models'

def set_dir(path):
    """test"""
    global root
    root = path

def get_dir():
    global root
    return root

def save(model, path, overwrite = False):
    global root
    os.makedirs(root, exist_ok = True)
    full_path = os.path.join(root, path)
    if os.path.exists(full_path):
        if overwrite:
            raise Exception(f'"{full_path}" already exists!')
    
    joblib.dump(model, full_path)

def load(path):
    global root
    full_path = os.path.join(root, path)
    if os.path.exists(full_path):
        return joblib.load(path)
    raise Exception(f'"{full_path}" does not exist!')

def grid_predictions(dataloader, Model, selection, num_features, dtype, common_kwargs = {}, min_count = 0, target_subset = None, metric = 'recall', mean_func = 'weighted_mean', **kwargs):

    values  = [num_features] + list(kwargs.values())
    params  = list(product(*values))
    names   = ['num_features'] + list(kwargs)
    C       = [len(num_features)] + [len(values) for values in kwargs.values()]
    M       = len(C)
    V       = [None] * len(names)
    
    with Waiting('loading', 'loaded', 'train / val data', inline = True):
        X_train, Y_train, X_val, Y_val = dataloader.load_train_val(features = selection[:max(num_features)], dtype = dtype, min_count = min_count, target_subset = target_subset)

    keys    = ['predict', 'predict_proba', 'predict_log_proba']
    outputs = {'target' : dataloader.decode(Y_val), 'labels' : list(dataloader._encoder), 'time' : [], 'num_features' : num_features}
    for key in keys:
        if hasattr(Model, key):
            outputs[key] = []

    best = (None, None, -1)
    for k, param in enumerate(params):
        for i, theta in enumerate(param):
            flag = V[i] != theta
            if (i + 1) < M and V[i] is not None and V[i] != theta:
                flag   = True
                delete = sum(C[i + 1:])
                for j in range(i + 1, M):
                    V[j] = None
            else:
                delete = 0
        
            V[i]    = theta
            message = ' '.join([f'{p} {v}' for p, v in zip(names[:i + 1], V[:i + 1])])

            if flag:
                msg(message, delete = delete)
        
        model = Model(**common_kwargs, **dict(zip(names[1:], V[1:])))

        start = time()
        model.fit(X_train[:,:param[0]], Y_train)
        fit   = time()
        hat   = model.predict(X_val[:,:param[0]])
        pred  = time()

        outputs['time'].append((fit - start, pred - fit))

        for key in keys:
            if hasattr(model, key):
                outputs[key].append(getattr(model, key)(X_val[:,:param[0]]))

        score = Metrics(Y_val, hat, metric)(func = mean_func)
        if score > best[2]:
            best_kwargs = {**common_kwargs, **dict(zip(names[1:], V[1:])), 'num_features' : param[0]}
            best        = (model, k, score)

        monitor_RAM()
    
    k                      = best[1]
    best                   = (best[0], dataloader.decode(outputs['predict'][k]))

    if hasattr(Model, 'predict_proba'):
        best              += (outputs['predict_proba'][k],)

    outputs['identifiers'] = dataloader.identifiers_val
    outputs['predict']     = dataloader.decode(np.array(outputs['predict']).reshape(*C, -1))
    
    for key in keys[1:]:
        if hasattr(Model, key):
            outputs[key]   = np.array(outputs[key]).reshape(*C, -1, len(dataloader._encoder))

    outputs['time']        = np.array(outputs['time']).reshape(*C, 2)
        
    outputs.update(common_kwargs)
    outputs.update(kwargs)

    outputs['best']   = best

    msg('computed predictions and computation times', delete = sum(C))
    
    return outputs, best_kwargs, np.vstack([X_train, X_val]), np.concatenate([Y_train, Y_val])