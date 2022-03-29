from   genolearn.utils import create_log
from   genolearn.dataloader import DataLoader
from   genolearn.models.classification import RandomForestClassifier
from   genolearn.logger import msg

from   time import time

import numpy as np
import os

msg('executing demo-A-rf.py')

dataloader = DataLoader('data', 'raw-data/meta-data.csv', 'Accession', 'Region', 'Year')

fisher = dataloader.load_feature_selection('fisher-score.npz')
orders = fisher.rank()

kwargs = dict(class_weight = 'balanced', n_jobs = -1)

K           = [100, 1000, 10000, 100000, 1000000]
max_depths  = range(5, 101, 5)

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

        predictions_depth = []
        times_depth       = []

        for max_depth in max_depths:
            
            msg(f'{year} {k:7d} {max_depth:2d}')
            predictions_seed = []
            times_seed       = []

            for seed in range(10):
                train = time()
                model = RandomForestClassifier(max_depth = max_depth, random_state = seed, **kwargs)
                model.fit(X_train[:,:k], Y_train)
                train = time() - train
                test  = time()
                pred  = model.predict(X_test[:,:k])
                test  = time() - test
                predictions_seed.append(dataloader.decode(pred))
                times_seed.append((train, test))

            predictions_depth.append(predictions_seed)
            times_depth.append(times_seed)

        msg('', inline = True, delete = len(max_depths))

        predictions_k.append(predictions_depth)
        times_k.append(times_depth)

    msg('', inline = True, delete = len(K))

    predictions.append(predictions_k)
    times.append(times_k)

outdir = 'script-output'
os.makedirs(outdir, exist_ok = True)

np.savez_compressed(f'{outdir}/random-forest.npz', predictions = predictions, times = times, K = K, max_depths = max_depths)

create_log(outdir, 'demo-A-rf.txt')

msg('executed demo-A-rf.py')
