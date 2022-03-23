from genolearn import DataLoader
from genolearn.models.classification import RandomForestClassifier
from genolearn.logger import msg

import numpy as np

dataloader = DataLoader('data-low-memory', 'raw-data/meta-data.csv', 'Accession', 'Region', 'Year')

fisher = dataloader.load_feature_selection('fisher-score.npz')
orders = fisher.rank()

K           = [100, 1000, 10000, 100000, 1000000]
max_depths  = range(5, 51, 5)

predictions = []

for year in reversed(range(2014, 2019)):

    features         = orders[str(year)][:max(K)]
    X_train, Y_train, X_test, Y_test = dataloader.load_train_test(range(year, 2019), [2019], features = features)
    
    predictions_k = []

    for k in K:

        predictions_depth = []

        for max_depth in max_depths:
            
            msg(f'{year} {k:7d} {max_depth:2d}', inline = True)
            predictions_seed = []

            for seed in range(10):
                model = RandomForestClassifier(max_depth = max_depth, random_state = seed, class_weight = 'balanced')
                model.fit(X_train, Y_train)
                predictions_seed.append(model.predict(X_test))

            predictions_depth.append(predictions_seed)

        predictions_k.append(predictions_depth)

    predictions.append(predictions_k)

    msg(year)
    
np.savez_compressed('random-forest.npz', predictions = predictions, K = K, max_depths = max_depths)
