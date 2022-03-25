from   genolearn import DataLoader, utils
from   genolearn.models.classification import LogisticRegression
from   genolearn.logger import msg

import numpy as np
import os

msg('executing demo-A-lr.py')

dataloader = DataLoader('data-low-memory', 'raw-data/meta-data.csv', 'Accession', 'Region', 'Year')

fisher = dataloader.load_feature_selection('fisher-score.npz')
orders = fisher.rank()

kwargs = dict(class_weight = 'balanced', n_jobs = -1)

K           = [100, 1000, 10000, 100000, 1000000]
C           = [1e-2, 1e-1, 1e0, 1e1, 1e2]

predictions = []

for year in reversed(range(2014, 2019)):

    msg(year)

    features         = orders[str(year)][:max(K)]
    X_train, Y_train, X_test, Y_test = dataloader.load_train_test(range(year, 2019), [2019], features = features)
    
    predictions_k = []

    for k in K:

        msg(f'{year} {k:7d}')

        predictions_depth = []

        for c in C:
            
            msg(f'{year} {k:7d} {c:2d}')
            predictions_seed = []

            for seed in range(10):
                model = LogisticRegression(C = c, random_state = seed, **kwargs)
                model.fit(X_train[:,:k], Y_train)
                predictions_seed.append(model.predict(X_test))

            predictions_depth.append(predictions_seed)

        msg('', inline = True, delete = len(C))

        predictions_k.append(predictions_depth)

    msg('', inline = True, delete = len(K))

    predictions.append(predictions_k)
    
outdir = 'script-output'
os.makedirs(outdir, exist_ok = True)

np.savez_compressed(f'{outdir}/logistic-regression.npz', predictions = predictions, K = K, C = C)

utils.create_log(outdir, 'demo-A-lr.txt')

msg('executed demo-A-lr.py')
