from   genolearn import DataLoader, utils
from   genolearn.models.classification import RandomForestClassifier
from   genolearn.logger import msg

import numpy as np
import os

msg('executing demo-A-rf.py')

dataloader = DataLoader('data-low-memory', 'raw-data/meta-data.csv', 'Accession', 'Region', 'Year')

fisher = dataloader.load_feature_selection('fisher-score.npz')
orders = fisher.rank()

kwargs = dict(class_weight = 'balanced', n_jobs = -1)

K           = [100, 1000, 10000, 100000, 1000000]
max_depths  = range(5, 51, 5)

predictions = []

for year in reversed(range(2014, 2019)):

    msg(year)

    features         = orders[str(year)][:max(K)]
    X_train, Y_train, X_test, Y_test = dataloader.load_train_test(range(year, 2019), [2019], features = features)
    
    predictions_k = []

    for k in K:

        msg(f'{year} {k:7d}')

        predictions_depth = []

        for max_depth in max_depths:
            
            msg(f'{year} {k:7d} {max_depth:2d}')
            predictions_seed = []

            for seed in range(10):
                model = RandomForestClassifier(max_depth = max_depth, random_state = seed, **kwargs)
                model.fit(X_train[:,:k], Y_train)
                predictions_seed.append(model.predict(X_test))

            predictions_depth.append(predictions_seed)

        msg('', inline = True, delete = len(max_depths))

        predictions_k.append(predictions_depth)

    msg('', inline = True, delete = len(K))

    predictions.append(predictions_k)
    
outdir = 'script-output'
os.makedirs(outdir, exist_ok = True)

np.savez_compressed(f'{outdir}/random-forest.npz', predictions = predictions, K = K, max_depths = max_depths)

utils.create_log(outdir, 'demo-A-rf.txt')

msg('executed demo-A-rf.py')
