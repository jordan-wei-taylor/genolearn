from   genolearn import working_directory
import os
import json
import scipy.sparse
import numpy  as np
import gzip

class DataLoader():
    """
    DataLoader Class

    Parameters
    ----------
        meta_file   : str
            Preprocessed metadata within ``preprocess_dir``/meta
        working_dir : str
            Path to working directory

    """
    def __init__(self, meta_file, working_dir = working_directory):
        
        if working_dir is None:
            raise Exception('execute GenoLearn\'s setup command first')

        if not os.path.exists(working_dir) or '.genolearn' not in os.listdir(working_dir):
            raise Exception(f'"{working_dir}" not a valid working directory')

        path  = os.path.join(working_dir, 'meta')
        if not os.path.exists(path):
            raise Exception('no preprocessed metadata to load')
        
        valid = os.listdir(path)

        if len(valid) == 0:
            raise Exception(f'no preprocessed metadata in {path}')
            
        if meta_file not in valid:
            raise Exception(f'"{meta_file}" not found in "{path}" - expect one of' + '\n  •'.join([''] + valid))

        self.dense          = os.path.exists(os.path.join(working_dir, 'preprocess', 'dense'))
        self.working_dir    = working_dir
        self.preprocess_dir = os.path.join(working_dir, 'preprocess')
        self.data_dir       = os.path.join(self.preprocess_dir, 'dense' if self.dense else 'sparse')
        
        with open(os.path.join(working_dir, 'meta', meta_file)) as f:
            self.meta = json.load(f)
            self.c    = len(set(self.meta['targets']))

        with open(os.path.join(self.preprocess_dir, 'info.json')) as f:
            info = json.load(f)
            self.n, self.m = info['n'], info['m']

    def _load_X(self, identifier, features):

        npz = os.path.join(self.data_dir, f'{identifier}.npz')

        if self.dense:
            arr, = np.load(npz).values()
        else:
            arr  = scipy.sparse.load_npz(npz)

        if features is not None:
            if self.dense:
                arr = arr[features]
            else:
                arr = arr[:,features]
        
        return arr

    def _load_Y(self, identifier):
        return self.meta['search'][identifier]

    def _check_identifiers(self, identifiers):
        ret = []
        
        for identifier in identifiers:

            # unlabelled data
            if identifier == 'unlabelled':
                for identifier in os.listdir(self.data_dir):
                    identifier = identifier.replace('.npz', '')
                    if identifier not in self.meta['identifiers']:
                        ret.append(identifier)
                    
            # check if it is a group and append entire associated identifiers
            elif identifier in self.meta['group']:
                ret += self.meta['group'][identifier]
            
            # check if it is Train / Test and append entire associated identifiers
            elif identifier in ['Train', 'Test']:
                for group in self.meta[identifier]:
                    ret += self.meta['group'][group]

            # append the identifier as is
            else:
                ret.append(identifier)

        return ret

    def load_X(self, *identifiers, features = None, dtype = np.uint16):
        r"""
        loads all observations with associated ``identifiers``. If ``features`` is provided, loads only
        those feature values. If ``dtype`` is provided, tries to convert the ``dtype`` provided.
        Returns
        -------
            X : numpy.ndarray or scipy.sparse.csr_matrix
                Defining :math:`n = |\text{identifiers}|`  and :math:`m` as the number of genome
                sequences identified during the preprocessing stage or :math:`|\text{features}|` if ``features``
                was provided, then, :math:`X\in\mathbb{Z}^{n,m}`. 
        """
        identifiers = self._check_identifiers(identifiers)
        m           = self.m if features is None else len(features)
        if self.dense:
            X = np.zeros((len(identifiers), m), dtype = dtype)
            for i, identifier in enumerate(identifiers):
                X[i] = self._load_X(identifier, features)
        else:
            X = scipy.sparse.vstack([self._load(identifier, features) for identifier in identifiers], dtype = dtype)
        return X

    def load_Y(self, *identifiers):
        """
        Returns
        -------
        Y : numpy.ndarray
        """
        identifiers = self._check_identifiers(identifiers)
        return np.array(list(map(self._load_Y, identifiers)))

    def load_train_test_identifiers(self, min_count = 0, target_subset = None):
        """
        loads train and test identifiers.
        Returns
        -------
            train_identifiers : numpy.ndarray
            test_identifiers  : numpy.ndarray
        """
        train   = self._check_identifiers(self.meta['Train'])
        test    = self._check_identifiers(self.meta['Test'])

        y_train = self.load_Y(*train)
        y_test  = self.load_Y(*test)

        unique, arg = np.unique(y_train, return_inverse = True)
        dummies     = np.eye(len(unique))[arg]

        label_counts = dummies.sum(axis = 0)
        labels       = unique[label_counts >= min_count]

        if target_subset:
            labels = [label for label in labels if label in target_subset]
        
        self._encoder = {label : i for i, label in enumerate(labels)}

        identifiers_train = np.array(train)[np.isin(y_train, labels)]
        identifiers_test  = np.array(test )[np.isin(y_test , labels)]

        return identifiers_train, identifiers_test

    def load_train_test(self, features = None, min_count = 0, target_subset = None, dtype = np.uint16):
        """
        using the method ``load_train_test_identifiers`` returns train and test data for supervised learning.
        Returns
        -------
            X_train : load_X(train_identifiers, features = features, sparse = sparse)
            Y_train : load_Y(train_identifiers)
            X_test  : load_X(test_identifiers, features = features, sparse = sparse)
            Y_test  : load_Y(test_identifiers)
        """
        self.identifiers_train, self.identifiers_test = self.load_train_test_identifiers(min_count, target_subset)

        Y_train = self.encode(self.load_Y(*self.identifiers_train))
        Y_test  = self.encode(self.load_Y(*self.identifiers_test ))

        X_train = self.load_X(*self.identifiers_train, features = features, dtype = dtype)
        X_test  = self.load_X(*self.identifiers_test , features = features, dtype = dtype)

        return X_train, Y_train, X_test, Y_test
    
    def encode(self, Y):
        return np.vectorize(lambda value : self._encoder[value])(Y)

    def decode(self, Y):
        decoder = {value : key for key, value in self._encoder.items()}
        return np.vectorize(lambda value : decoder[value])(Y)

    def features(self, indices = None):
        if '_features' not in self.__dict__:
            with gzip.open(os.path.join(self.preprocess_dir, 'features.txt.gz')) as g:
                self._features = g.read().decode().split()
        return self._features if indices is None else [self._features[i] for i in indices]

    def load_feature_selection(self, file):
        ret, = np.load(os.path.join(self.working_dir, 'feature-selection', file), allow_pickle = True).values()
        return ret

    def generator(self, *identifiers, features = None, force_dense = False, force_sparse = False):
        """
        Iteratively yields an x, y pair from the method ``load``.
        """
        identifiers = self._check_identifiers(identifiers)

        for identifier in identifiers:
            X, Y = self._load_X(identifier, features), self.load_Y(identifier)[0]
            if force_sparse:
                if isinstance(X, np.ndarray):
                    X = scipy.sparse.csr_matrix(X.reshape(1, -1))
            if force_dense:
                if not isinstance(X, np.ndarray):
                    X = X.A.flatten()
            yield X, Y