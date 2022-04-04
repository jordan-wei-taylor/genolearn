from   genolearn.logger import msg, Waiting
from   genolearn.models import grid_predictions
from   genolearn.dataloader import DataLoader
from   genolearn.utils import create_log

from   sklearn.ensemble import RandomForestClassifier

import numpy as np
import os

script = 'demo-A-rf'

msg(f'executing {script}.py')

dataloader = DataLoader('data', 'raw-data/meta-data.csv', 'Accession', 'Region', 'Year')

with Waiting('loading', 'loaded', 'fisher scores', inline = True):
    orders = dataloader.load_feature_selection('fisher-score.npz').rank()

test       = [2019]

K          = [100, 1000, 10000, 100000, 1000000]

common     = dict(n_jobs = -1, class_weight = 'balanced')
params     = dict(max_depth = range(5, 101, 5), random_state = range(10))

Hats       = []
Times      = []

for year in reversed(range(2014, 2019)):
    train = range(year, 2019)
    order = orders[str(year)]
    msg(f'train = {train}\ttest = {test}')
    hats, times = grid_predictions(dataloader, train, test, RandomForestClassifier, K, order, common, **params)

    Hats.append(hats)
    Times.append(times)

path = 'script-output'
file = f'{script}.npz'
full = os.path.join(path, file)

os.makedirs(path, exist_ok = True)

with Waiting('generating', 'generated', full, inline = True):
    np.savez(full, hats = Hats, times = Times, K = K, **params)

create_log(path, f'{script}-log.txt')

msg(f'executed {script}.py')

