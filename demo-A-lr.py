from   genolearn.utils import create_log
from   genolearn.dataloader import DataLoader
from   genolearn.models.classification import LogisticRegression
from   genolearn.logger import msg

from   time import time

import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

msg('executing demo-A-lr.py')

dataloader = DataLoader('data', 'raw-data/meta-data.csv', 'Accession', 'Region', 'Year')

orders = dataloader.load_feature_selection('fisher-score.npz').rank()

kwargs = dict(class_weight = 'balanced', solver = 'saga', n_jobs = -1)

K           = [100, 1000, 10000, 100000, 1000000][:-1] # k = 1,000,000 takes far too long
C           = [1e-2, 1e0, 1e2]

predictions = []
times       = []

for year in reversed(range(2014, 2019)):

    msg(year)

    features         = orders[str(year)][:max(K)]
    X_train, Y_train, X_test, Y_test = dataloader.load_train_test(range(year, 2019), [2019], features = features)
    
    predictions_k = []
    times_k       = []

    for k in K:

        msg(f'{year} {k:7d}')

        predictions_c = []
        times_c       = []

        for c in C:
            
            msg(f'{year} {k:7d} {c:.1e}')
            predictions_seed = []
            times_seed       = []

            for seed in range(10):
                train = time()
                model = LogisticRegression(C = c, random_state = seed, **kwargs)
                model.fit(X_train[:,:k], Y_train)
                train = time() - train
                test  = time()
                pred  = model.predict(X_test[:,:k])
                test  = time() - test
                predictions_seed.append(dataloader.decode(pred))
                times_seed.append((train, test))

            predictions_c.append(predictions_seed)

        msg('', inline = True, delete = len(C))

        predictions_k.append(predictions_c)

    msg('', inline = True, delete = len(K))

    predictions.append(predictions_k)
    
outdir = 'script-output'
os.makedirs(outdir, exist_ok = True)

np.savez_compressed(f'{outdir}/logistic-regression.npz', predictions = predictions, K = K, C = C)

create_log(outdir, 'demo-A-lr.txt')

msg('executed demo-A-lr.py')
