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
            self.n = d['n']
            self.m = d['m']
            
        self.c = len(set(df[target]))

    def _check_path(self, identifier, sparse):
        npz = os.path.join(self._sparse if sparse else self._dense, f'{identifier}.npz')
        if os.path.exists(npz):
            return npz
        # raise Exception(f'"{npz}" not a valid path!')

    def _check_meta(self, *identifiers, column = None):
        if column is None:
            column = self.target
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
                arr  = scipy.sparse.load_npz(npz)
            else:
                arr, = np.load(npz).values()
        except:
            raise Exception(f'"{os.path.join(self._sparse if sparse else self._dense)}" does not exist!')

        if features is not None:
            arr = arr[:,features]

        return arr

    def get_identifiers(self, *values, column):
        self._check_meta(*values, column = column)            
        identifiers = self.meta.index[self.meta[column].isin(values)].values
        return identifiers

    def load_X(self, *identifiers, features = None, sparse = None):
        if f'{identifiers[0]}.npz' in os.listdir(self._sparse if sparse else self._dense):
            npzs = [self._check_path(identifier, sparse) for identifier in identifiers]
            X    = [self._load_X(npz, features, sparse) for npz in npzs]
            return scipy.sparse.vstack(X) if sparse else np.array(X)
        else:
            identifiers = self.get_identifiers(*identifiers, column = self.group)
            return self.load_X(*identifiers)

    def load_Y(self, *identifiers):
        return self._check_meta(*identifiers)

    def load(self, *identifiers):
        return self.load_X(*identifiers), self.load_Y(*identifiers)

    def generator(self, *identifiers, features = None, sparse = None):
        for identifier in identifiers:
            if f'{identifier}.npz' in os.listdir(self._sparse if sparse else self._dense):
                npz = self._check_path(identifier, sparse)
                yield self._load_X(npz, features, sparse), self.load_Y(identifier)
            elif identifier in self.meta[self.group].values:
                yield from self.generator(*self.get_identifiers(identifier, column = self.group))
