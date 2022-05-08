from   genolearn.models import classification
from   genolearn.logger import Waiting, msg
from   genolearn.utils  import monitor_RAM

from   itertools        import product
from   time             import time

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

def grid_predictions(dataloader, train, test, Model, K, order = None, common_kwargs = {}, min_count = 0, target_subset = None, **kwargs):

    values  = [K] + list(kwargs.values())
    params  = list(product(*values))
    names   = ['k'] + list(kwargs)
    C       = [len(K)] + [len(values) for values in kwargs.values()]
    M       = len(C)
    V       = [None] * len(names)
    
    with Waiting('loading', 'loaded', 'train / test data', inline = True):
        X_train, Y_train, X_test, Y_test = dataloader.load_train_test(train, test, features = order[:max(K)], min_count = min_count, target_subset = target_subset)

    keys    = ['predict_proba', 'predict_log_proba']
    outputs = {'labels' : list(dataloader.encoder), 'time' : [], 'predict' : [], 'K' : K}
    for key in keys:
        if hasattr(Model, key):
            outputs[key] = []

    for param in params:
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
        hat   = model.predict(X_test[:,:param[0]])
        pred  = time()
        outputs['predict'].append(hat)
        outputs['time'].append((fit - start, pred - fit))

        for key in keys:
            if hasattr(model, key):
                outputs[key].append(getattr(model, key)(X_test[:,:param[0]]))

        monitor_RAM()

    outputs['predict'] = np.array(outputs['predict']).reshape(*C, -1)
    outputs['time']    = np.array(outputs['time']).reshape(*C, 2)

    if len(params) == 1:
        outputs['model'] = model
        
    outputs.update(common_kwargs)
    outputs.update(kwargs)

    msg('computed predictions and computation times', delete = sum(C))
    
    return outputs