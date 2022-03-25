import scipy.sparse

import numpy  as np
import pandas as pd

import json
import os

class Dict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rank(self, ascending = False):
        ret = {}
        for key in self:
            rank     = self[key].argsort(axis = -1)
            ret[key] = rank if ascending else rank[::-1]
        return ret

class DataLoader():

    def __init__(self, path, meta_path, identifier = None, target = None, group = None, sparse = False):
        self.path       = path
        self.meta_path  = path
        self.identifier = identifier
        self.target     = target
        self.group      = group
        self.sparse     = sparse

        self._sparse   = os.path.join(path, 'sparse')
        self._dense    = os.path.join(path, 'dense')

        self.valid     = set()

        if os.path.exists(self._sparse):
            self.valid |= set(npz.replace('.npz', '') for npz in os.listdir(self._sparse) if npz.endswith('.npz'))

        if os.path.exists(self._dense):
            self.valid |= set(npz.replace('.npz', '') for npz in os.listdir(self._dense ) if npz.endswith('.npz'))

        df = pd.read_csv(meta_path)
        if identifier is not None:
            df = df.set_index(identifier)
            self.valid = set(self.valid) & set(df.index)

        if group:
            df[group] = df[group].apply(str)

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
        identifiers = [str(identifier) for identifier in identifiers]
        if column is None:
            column = self.group
        if self.meta is None:
            raise Exception('Meta data not loaded! Run the load_meta method first!')
        if column and column not in self.meta.columns:
            raise Exception(f'"{column}" not a valid column in self.meta!')
        if identifiers and self.meta.index.isin(identifiers).any():
            return self.meta.loc[identifiers, self.target]
        if identifiers and column and self.meta[column].isin(identifiers).any():
            return self.meta.loc[self.meta[column].isin(identifiers),self.target]
        raise Exception()
        
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
            if sparse:
                arr = arr[:,features]
            else:
                arr = arr[features]

        return arr

    def get_identifiers(self, *values, column):
        values = [str(value) for value in values]
        self._check_meta(*values, column = column)            
        identifiers = self.meta.index[self.meta[column].isin(values)].values
        return identifiers

    def load_X(self, *identifiers, features = None, sparse = None):
        self._identifiers = identifiers
        self._features    = features
        if f'{identifiers[0]}.npz' in os.listdir(self._sparse if sparse else self._dense):
            npzs = [self._check_path(identifier, sparse) for identifier in identifiers]
            X    = [self._load_X(npz, features, sparse) for npz in npzs]
            return scipy.sparse.vstack(X) if sparse else np.array(X)
        else:
            identifiers = self.get_identifiers(*identifiers, column = self.group)
            return self.load_X(*identifiers, features = features, sparse = sparse)

    def load_Y(self, *identifiers):
        Y = self._check_meta(*identifiers)
        return np.array(Y).flatten()[0] if len(Y) == 1 else Y

    def load(self, *identifiers, features = None, sparse = None):
        return self.load_X(*identifiers, features = features, sparse = sparse), self.load_Y(*identifiers)

    def load_train_test_identifiers(self, train_identifiers, test_identifiers, min_count = 0):

        y_train           = self.load_Y(*train_identifiers)
        y_test            = self.load_Y(*test_identifiers)


        dummy             = pd.get_dummies(y_train)
        label_counts      = dummy.sum()
        labels            = label_counts.index[label_counts >= min_count]

        self._encoder     = {label : i for i, label in enumerate(labels)}

        train_mask        = y_train.isin(self._encoder)
        test_mask         = y_test.isin(self._encoder)

        train_identifiers = train_mask.index[train_mask]
        test_identifiers  = test_mask.index[test_mask]

        return train_identifiers, test_identifiers
    
    def load_train_test(self, train_identifiers, test_identifiers, features = None, sparse = None, min_count = 0):
        
        identifiers       = self.load_train_test_identifiers(train_identifiers, test_identifiers, min_count)

        Y_train           = self.load_Y(*identifiers[0]).apply(lambda target : self._encoder[target]).values
        Y_test            = self.load_Y(*identifiers[1]).apply(lambda target : self._encoder[target]).values

        X_train           = self.load_X(*train_identifiers, features = features, sparse = sparse)
        X_test            = self.load_X(*test_identifiers , features = features, sparse = sparse)

        self._identifiers = train_identifiers, test_identifiers

        return X_train, Y_train, X_test, Y_test

    def generator(self, *identifiers, features = None, sparse = None, force_dense = False, force_sparse = False):
        for identifier in identifiers:
            if f'{identifier}.npz' in os.listdir(self._sparse if sparse else self._dense):
                npz  = self._check_path(identifier, sparse)
                X, Y = self._load_X(npz, features, sparse), self.load_Y(identifier)
                if force_sparse:
                    if isinstance(X, np.ndarray):
                        X = scipy.sparse.csr_matrix(X.reshape(1, -1))
                if force_dense:
                    if not isinstance(X, np.ndarray):
                        X = X.A.flatten()
                yield X, Y
            elif identifier in self.meta[self.group].values:
                yield from self.generator(*self.get_identifiers(identifier, column = self.group))

    @property
    def identifiers(self):
        return self._identifiers

    @property
    def features(self):
        return self._features

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return {value : key for key, value in self.encoder.items()}

    def load_feature_selection(self, file):
        npz = np.load(os.path.join(self.path, 'feature-selection', file), allow_pickle = True)
        return Dict(npz)
