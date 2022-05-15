from   genolearn import feature_selection, dataloader, metrics
from   genolearn import models
from   genolearn import utils

import numpy     as np
import os

Dataloader = dataloader.DataLoader

def feature_selection(function, data_path, meta_path, target_column, index_column, group_column = None):
    dataloader = dataloader.DataLoader(data_path, meta_path, index_column)
    npzs       = [npz.replace('.npz', '') for npz in os.listdir(data_path) if npz.endswith('.npz')]
    valid_ix   = dataloader.meta.index.isin(npzs)
    meta       = dataloader.meta.loc[valid_ix].reset_index()

    function(dataloader, meta, target_column, index_column, group_column)
