from   genolearn._base import Dict

import scipy.sparse

import numpy  as np
import pandas as pd

import json
import os

class DataLoader():
    """
    DataLoader Class

    Parameters
    ----------
        path : str
            Path to directory of preprocessed data.

        meta_path : str
            Path to meta data.

        identifier : str
            Column name within the meta data denoting the unique identifiers.
        
        target : str
            Column name within the meta data denoting the target.
        
        group : str, *default=None*
            Column name within the meta data denoting how the data may be grouped.

        sparse : bool, *default=False*
            Identify if preprocessed data is in sparse format.
        
    """
    
    def __init__(self, path, meta_path, identifier, target, group = None, sparse = False):
        self.path       = path
        self.meta_path  = meta_path
        self.identifier = identifier
        self.target     = target
        self.group      = group
        self.sparse     = sparse
        
        self._sparse    = os.path.join(path, 'sparse')
        self._dense     = os.path.join(path, 'dense')

        df = pd.read_csv(meta_path)
        if identifier is not None:
            df = df.set_index(identifier)

        if group:
            df[group] = df[group].apply(str)

        self.meta  = df

        with open(os.path.join(path, 'meta.json')) as f:
            d      = json.load(f)
            self.n = d['n']
            self.m = d['m']
            
        self.c = len(set(df[target]))

    def _check_path(self, identifier, sparse):
        """
        Checks if the identifier is valid

        Returns
        -------
            npz : str or None
                If a valid identifier, path to ``.npz`` file associated with ``identifier``,
                otherwise, ``None``.
        """
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

    def _get_identifiers(self, *values, column):
        values = [str(value) for value in values]
        self._check_meta(*values, column = column)            
        identifiers = self.meta.index[self.meta[column].isin(values)].values
        return identifiers

    def load_X(self, *identifiers, features = None, sparse = None):
        r"""
        Loads all observations with associated ``identifiers``. If ``features`` is provided, loads only
        those feature values. If ``sparse`` is provided, override the original ``sparse`` setting when
        the class object was instantiated.

        Returns
        -------
            X : numpy.ndarray or scipy.sparse.csr_matrix
                Defining :math:`n = |\text{identifiers}|`  and :math:`m` as the number of genome
                sequences identified during the preprocessing stage or :math:`|\text{features}|` if ``features``
                was provided, then, :math:`X\in\mathbb{Z}^{n,m}`. If ``sparse`` is ``False``, return an ndarray,
                if ``sparse`` is ``True``, return a csr_matrix, otherwise, assume the ``sparse`` setting from
                the __init__.

        """
        self._identifiers = identifiers
        self._features    = features
        if f'{identifiers[0]}.npz' in os.listdir(self._sparse if sparse else self._dense):
            npzs = [self._check_path(identifier, sparse) for identifier in identifiers]
            X    = [self._load_X(npz, features, sparse) for npz in npzs]
            return scipy.sparse.vstack(X) if sparse else np.array(X)
        else:
            identifiers = self._get_identifiers(*identifiers, column = self.group)
            return self.load_X(*identifiers, features = features, sparse = sparse)

    def load_Y(self, *identifiers):
        """
        Returns
        -------
            Y : str or pandas.Series
        """
        Y             = self._check_meta(*identifiers)
        return np.array(Y).flatten()[0] if len(Y) == 1 else Y

    def load(self, *identifiers, features = None, sparse = None):
        """
        Returns
        -------
            X : load_X(\*identifiers, features = features, sparse = sparse)
            Y : load_Y(\*identifiers)
        """
        return self.load_X(*identifiers, features = features, sparse = sparse), self.load_Y(*identifiers)

    def load_train_test_identifiers(self, train_identifiers, test_identifiers, min_count = 0, target_subset = None):
        """
        Identifiers which of the ``test_identifiers``' targets are also in the ``train_identifiers``' targets
        only counting training targets that have a count of at least ``min_count``.

        Returns
        -------
            train_identifiers : pandas.Index
            test_identifiers : pandas.Index

        """
        y_train           = self.load_Y(*train_identifiers)
        y_test            = self.load_Y(*test_identifiers)

        dummy             = pd.get_dummies(y_train)
        label_counts      = dummy.sum()
        labels            = label_counts.index[label_counts >= min_count]

        if target_subset:
            labels = [label for label in labels if label in target_subset]

        self._encoder     = {label : i for i, label in enumerate(labels)}

        train_mask        = y_train.isin(self._encoder)
        test_mask         = y_test.isin(self._encoder)

        train_identifiers = train_mask.index[train_mask]
        test_identifiers  = test_mask.index[test_mask]

        return train_identifiers, test_identifiers
    
    def load_train_test(self, train_identifiers, test_identifiers, features = None, sparse = None, min_count = 0, target_subset = None):
        """
        Using the method ``load_train_test_identifiers`` returns train and test data for supervised learning.

        Returns
        -------
            X_train : load_X(train_identifiers, features = features, sparse = sparse)
            Y_train : load_Y(train_identifiers)
            X_test  : load_X(test_identifiers, features = features, sparse = sparse)
            Y_test  : load_Y(test_identifiers)
        """
        identifiers       = self.load_train_test_identifiers(train_identifiers, test_identifiers, min_count, target_subset)

        Y_train           = self.encode(self.load_Y(*identifiers[0]))
        Y_test            = self.encode(self.load_Y(*identifiers[1]))

        X_train           = self.load_X(*identifiers[0], features = features, sparse = sparse)
        X_test            = self.load_X(*identifiers[1] , features = features, sparse = sparse)

        self._identifiers = train_identifiers, test_identifiers

        return X_train, Y_train, X_test, Y_test

    def generator(self, *identifiers, features = None, sparse = None, force_dense = False, force_sparse = False):
        """
        Iteratively yields an x, y pair from the method ``load``.

        Yields
        ------
            x : load_X(identifier, features = features, sparse = sparse)
            y : load_Y(identifier)
        """
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
                yield from self.generator(*self._get_identifiers(identifier, column = self.group))

    @property
    def identifiers(self):
        """ The ``identifiers`` from the most recent call of ``load_X`` or ``load_train_test``.  """
        return self._identifiers

    def features(self, indices):
        """ Returns the features from `features.txt` at indices ``indices``  """
        with open(os.path.join(self.path, 'features.txt')) as f:
            features = f.read().split()
            return [features[i] for i in indices]

    @property
    def encoder(self):
        """ Encoding from integer to targets present in the meta-data (one-hot encoding). """
        if self._encoder:
            return self._encoder
        raise Exception('encoder is only available after running the `load_train_test` method!')

    def encode(self, Y):
        """ Returns an ``encoder`` look-up for every element in ``Y`` """
        if self._encoder:
            return np.vectorize(lambda value : self._encoder[value])(Y)
        raise Exception('encode is only available after running the `load_train_test` method!')

    @property
    def decoder(self):
        """ Decoder from targets present in the meta-data to integers. """
        if self._encoder:
            return {value : key for key, value in self.encoder.items()}
        raise Exception('decoder is only available after running the `load_train_test` method!')

    def decode(self, Y):
        """ Returns an ``decoder`` look-up for every element in ``Y`` """
        if self._encoder:
            decoder = self.decoder
            return np.vectorize(lambda value : decoder[value])(Y)
        raise Exception('decode is only available after running the `load_train_test` method!')

    def load_feature_selection(self, file):
        """
        Retrieves feature selection ``file`` from subdirectory "feature-selection".
        
        Returns
        -------
            features : Dict
                A dictionary of the form {key : numpy.ndarray } where the values of the dictionary
                are of shape :math:`(m,)` .
        """
        npz      = np.load(os.path.join(self.path, 'feature-selection', file), allow_pickle = True)
        features = Dict(npz)
        return features
