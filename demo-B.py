from   genolearn.dataloader import DataLoader
from   genolearn.models     import grid_predictions
from   genolearn.logger     import msg, Waiting
from   genolearn.utils      import create_log

from   sklearn.linear_model import LogisticRegression
from   sklearn.ensemble     import RandomForestClassifier

import warnings
import numpy as np
import os

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

script = 'demo-B'

msg(f'executing {script}')

dataloader = DataLoader('data', 'raw-data/meta-data.csv', 'Accession', 'Region', 'Year')

with Waiting('loading', 'loaded', 'fisher scores', inline = True):
    orders = dataloader.load_feature_selection('fisher-score.npz').rank()
    order  = orders['2014']

train  = range(2014, 2019)
test   = [2019]

K_rf   = [100, 1000, 10000, 100000, 1000000]
K_lr   = K_rf[:-1]

common_rf = dict(n_jobs = -1, class_weight = 'balanced')
common_lr = dict(n_jobs = -1, class_weight = 'balanced', solver = 'saga')

params_rf = dict(max_depth = range(5, 101, 5), random_state = range(10))
params_lr = dict(C = [1e-2, 1, 1e2], random_state = range(10))

rf = ['random-forest.npz', RandomForestClassifier, K_rf, common_rf, params_rf]
lr = ['logistic-regression.npz', LogisticRegression, K_lr, common_lr, params_lr]

for file, model, K, common, params in [rf, lr]:

    msg(f'computing contents for {file}')
    hats, times = grid_predictions(dataloader, train, test, model, K, order, common, **params)

    with Waiting('generating', 'generated', file):
        np.savez(os.path.join('script-output', file), hats = hats, times = times, K = K, **params)

create_log('script-output', 'demo-B-log.txt')

msg(f'executed {script}')
