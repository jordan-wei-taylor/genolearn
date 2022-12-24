from   genolearn.utils import working_directory
import os
import json
import numpy  as np
import gzip
import warnings

warnings.filterwarnings('ignore')

class DataLoader():

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

        self.working_dir    = working_dir
        self.preprocess_dir = os.path.join(working_dir, 'preprocess')
        self.data_dir       = os.path.join(self.preprocess_dir, 'array')
        
        with open(os.path.join(working_dir, 'meta', meta_file)) as f:
            self.meta = json.load(f)
            self.c    = len(set(self.meta['targets']))

        with open(os.path.join(self.preprocess_dir, 'info.json')) as f:
            info = json.load(f)
            self.n, self.m = info['n'], info['m']

    def _load_X(self, identifier, features, dtype):

        npz  = os.path.join(self.data_dir, f'{identifier}.npz')

        arr, = np.load(npz).values()

        if features is not None:
            arr = arr[features]
        
        return arr.astype(dtype) if dtype else arr

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
            
            # check if it is Train / Val and append entire associated identifiers
            elif identifier in ['Train', 'Val']:
                for group in self.meta[identifier]:
                    ret += self.meta['group'][group]

            # append the identifier as is
            else:
                ret.append(identifier)

        return ret

    def load_X(self, *identifiers, features = None, dtype = None):
        identifiers = self._check_identifiers(identifiers)
        features    = np.array(features)
        m           = self.m if features is None else features.sum() if features.dtype == np.bool_ else len(features)
        X           = np.zeros((len(identifiers), m), dtype = dtype)
        for i, identifier in enumerate(identifiers):
            X[i] = self._load_X(identifier, features, dtype)
        return X

    def load_Y(self, *identifiers):
        """
        Returns
        -------
        Y : numpy.ndarray
        """
        identifiers = self._check_identifiers(identifiers)
        return np.array(list(map(self._load_Y, identifiers)))

    def load_train_val_identifiers(self, min_count = 0, target_subset = None):
        train   = self._check_identifiers(self.meta['Train'])
        val     = self._check_identifiers(self.meta['Val'])

        y_train = self.load_Y(*train)
        y_val   = self.load_Y(*val)

        unique, arg = np.unique(y_train, return_inverse = True)
        dummies     = np.eye(len(unique))[arg]

        label_counts = dummies.sum(axis = 0)
        labels       = unique[label_counts >= min_count]

        if target_subset:
            labels = [label for label in labels if label in target_subset]
        
        self._encoder = {label : i for i, label in enumerate(labels)}

        identifiers_train = np.array(train)[np.isin(y_train, labels)]
        identifiers_val   = np.array(val  )[np.isin(y_val  , labels)]

        return identifiers_train, identifiers_val

    def load_train_val(self, features = None, dtype = None, min_count = 0, target_subset = None):
        self.identifiers_train, self.identifiers_val = self.load_train_val_identifiers(min_count, target_subset)

        Y_train = self.encode(self.load_Y(*self.identifiers_train))
        Y_val   = self.encode(self.load_Y(*self.identifiers_val  ))

        X_train = self.load_X(*self.identifiers_train, features = features, dtype = dtype)
        X_val   = self.load_X(*self.identifiers_val  , features = features, dtype = dtype)

        return X_train, Y_train, X_val, Y_val
    
    def load_identifiers(self, min_count = 0, target_subset = None):
        train   = self._check_identifiers(self.meta['Train'] + self.meta['Val'])

        y_train = self.load_Y(*train)

        unique, arg = np.unique(y_train, return_inverse = True)
        dummies     = np.eye(len(unique))[arg]

        label_counts = dummies.sum(axis = 0)
        labels       = unique[label_counts >= min_count]

        if target_subset:
            labels = [label for label in labels if label in target_subset]
        
        self._encoder = {label : i for i, label in enumerate(labels)}

        identifiers = np.array(train)[np.isin(y_train, labels)]

        return identifiers

    def load(self, features = None, dtype = None, min_count = 0, target_subset = None):

        self.identifiers = self.load_identifiers(min_count, target_subset)

        Y = self.encode(self.load_Y(*self.identifiers))
        X = self.load_X(*self.identifiers, features = features, dtype = dtype)

        return X, Y

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

    def generator(self, *identifiers, features = None, dtype = None):
        """
        Iteratively yields an x, y pair from the method ``load``.
        """
        identifiers = self._check_identifiers(identifiers)

        for identifier in identifiers:
            yield self._load_X(identifier, features, dtype), self.load_Y(identifier)[0]
