import scipy.sparse

import numpy  as np
import pandas as pd

import json
import os

class DataLoader():

    def __init__(self, path, meta_path, index = None, target = None, group = None, sparse = False):
        self.path      = path
        self.meta_path = path
        self.index     = index
        self.target    = target
        self.group     = group
        self.sparse    = sparse

        self._sparse   = os.path.join(path, 'sparse')
        self._dense    = os.path.join(path, 'dense')

        self.valid     = set()

        if os.path.exists(self._sparse):
            self.valid |= set(npz.replace('.npz', '') for npz in os.listdir(self._sparse) if npz.endswith('.npz'))

        if os.path.exists(self._dense):
            self.valid |= set(npz.replace('.npz', '') for npz in os.listdir(self._dense ) if npz.endswith('.npz'))

        df = pd.read_csv(meta_path)
        if index is not None:
            df = df.set_index(index)
            self.valid = set(self.valid) & set(df.index)

        self.valid = list(self.valid)
        self.meta  = df

        with open(os.path.join(path, 'meta.json')) as f:
            d = json.load(f)
            self.m = d['m']

    def _check_path(self, identifier):
        npz = os.path.join(self.path, identifier + '.npz')
        if os.path.exists(npz):
            return npz
        # raise Exception(f'"{npz}" not a valid path!')

    def _check_meta(self, *identifiers, column = None):
        if self.meta is None:
            raise Exception('Meta data not loaded! Run the load_meta method first!')
        if column not in self.meta.columns:
            raise Exception(f'"{column}" not a valid column in self.meta!')
        if identifiers and self.meta.index.isin(identifiers).any():
            return self.meta.loc[identifiers, column]
        if identifiers and self.meta[column].isin(identifiers).any():
            return self.meta.loc[self.meta[column].isin(identifiers),column]
        return Exception()
        
    def _load_X(self, npz, features, sparse = None):
        if sparse is None:
            sparse = self.sparse
        
        try:
            if sparse:
                arr = scipy.sparse.load_npz(os.path.join(self._sparse, f'{npz}.npz'))
            else:
                arr = np.load(os.path.join(self._dense, f'{npz}.npz')).reshape(-1, 1)
        except:
            raise Exception(f'"{os.path.join(self._sparse if sparse else self._dense)}" does not exist!')

        if features is not None:
            arr = arr[:,features]

        return arr

    def get_identifiers(self, *values, column):
        self._check_meta(*values, column = column)            
        identifiers = self.meta.index[self.meta[column].isin(values)].values
        return identifiers

    def load_X(self, *identifiers, features = None, column = None, sparse = None):
        if column is None:
            npzs = [self._check_path(identifier) for identifier in identifiers]
            X    = [self._load_X(npz, features, sparse) for npz in npzs]
            return scipy.sparse.vstack(X) if sparse else np.vstack(X)
        else:
            identifiers = self.get_identifiers(*identifiers, column = column)
            return self.load_X(*identifiers)

    def load_Y(self, *identifiers, column):
        return self._check_meta(*identifiers, column)

    def load(self, *identifiers, column):
        return self.load_X(*identifiers, column), self.load_Y(*identifiers, column)