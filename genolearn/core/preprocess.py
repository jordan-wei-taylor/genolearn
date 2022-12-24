import shutil
import numpy as np
import os
import re

clean_sample   = lambda sample : sample.replace('/', '.')
gather_feature = lambda line : line[:line.index(' ')]
gather_samples = lambda line : list(map(clean_sample, re.findall(r'(?<= )[^: ]+(?=:)', line)))
gather_counts  = lambda line : re.findall(r'(?<=:)[0-9]+', line)

def get_dtype(val):
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtypes:
        info = np.iinfo(dtype)
        if info.min <= val <= info.max:
            return dtype
    raise Exception()

def init(file, subpath = 'temp', ext = 'txt'):
    path = os.path.join(subpath, f'{file}.{ext}') if subpath else f'{file}.{ext}'
    if os.path.exists(path):
        os.remove(path)
    return open(path, 'a')

def add(object, i, count):
    object.write(f'{i} {count}\n')
    
def preprocess(preprocess_dir, data, batch_size, n_processes, max_features, verbose):
    """
    Preprocess a gunzip (gz) compressed text file containing genome sequence data of the following sparse format

        \b
        sequence_1 | identifier_{1,1}:count_{1,1} identifier_{1,1}:count_{2,1} ...
        sequence_2 | identifier_{2,1}:count_{2,1} identifier_{2,1}:count_{2,2} ...
        ...

    into a directory of .npz files, a list of all the features, and some meta information containing number of 
    identifiers, sequences, and non-zero counts.

    It is expected that the parameter `data` is in the `data_dir` directory set in the \033[1mactive config\033[0m file.
    See https://genolearn.readthedocs.io/tutorial/config for more details.

    \b\n
    Example Usage

    \b
    # reduce ram usage by setting a batch size (will increase preprocess time)
    >>> genolearn preprocess file.gz --batch-size 128
    """

    from   genolearn.logger       import msg, Waiting
    from   genolearn              import utils

    from   pathos.multiprocessing import cpu_count, Pool

    import resource
    import json
    import gzip

    try:
        # remove number of files open restriction
        limit = min(resource.getrlimit(resource.RLIMIT_NOFILE)[1], batch_size + 100)
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
    except:
        msg(f'Attempting to open to many files! Try reducing the number')
        return 

    if max_features == None:
        max_features = -1

    n_processes    = cpu_count() if n_processes == 'auto' else int(n_processes)
    
    first_run  = True
    features   = []
    exceptions = set()
    C          = 0
    hi         = 0
    unique     = set()

    if data.endswith('.gz'):
        _open  = gzip.GzipFile
        decode = lambda line : line.decode()
    else:
        _open  = open
        decode = lambda line : line

    with _open(os.path.expanduser(data)) as gz:
        
        if os.path.exists(preprocess_dir):                    
            shutil.rmtree(preprocess_dir)

        os.mkdir(preprocess_dir)
        os.chdir(preprocess_dir)

        if 'temp' in os.listdir():
            shutil.rmtree('temp')

        os.mkdir('temp')
        os.mkdir('array')

        files = {}

        while True:
           
            gz.seek(0)

            skip    = False
            skipped = False
            c       = 0
           
            for m, line in enumerate(gz, 1):

                line   = decode(line)

                srrs   = gather_samples(line)
                counts = gather_counts(line)

                if first_run:
                    features.append(gather_feature(line))
                    hi      = max(hi, *map(int, counts))
                    unique |= set(srrs)

                for SRR, count in zip(srrs, counts):
                    if SRR not in exceptions:
                        if SRR not in files and c <= batch_size:
                            if skip:
                                skipped = True
                                continue
                            files[SRR] = init(SRR)
                            c         += 1
                            C         += 1
                            skip = c == batch_size
                        add(files[SRR], m - 1, count)

                msg(f'{C:10,d} {m:10,d}', inline = m % verbose)                  
               
                if m == max_features:
                    break

            for f in files.values():
                f.close()

            if first_run:
                first_run = False

                d_dtype   = get_dtype(hi)
                c_dtype   = get_dtype(m)

                with Waiting('compressing', 'compressed', 'features.txt.gz'):
                    with gzip.open('features.txt.gz', 'wb') as g:
                        g.write(' '.join(features).encode())
                
                features.clear()

                f = init('info', None, 'json')
                json.dump({'n' : len(unique), 'm' : m, 'max' : hi}, f)
                f.close()                    
               
                def convert(file):
                    txt    = os.path.join('temp', f'{file}.txt')
                    npz    = f'{file}.npz'
                    c, d   = np.loadtxt(txt, dtype = c_dtype).T
                    arr    = np.zeros(m, dtype = d.dtype)
                    arr[c] = d
                    np.savez_compressed(os.path.join('array', npz), arr = arr)
                    os.remove(txt)
           
            with Waiting('converting', 'converted', 'to arrays'):
                with Pool(n_processes) as pool:
                    pool.map(convert, list(files))

            if not skipped:
                break

            exceptions |= set(files)

            if len(files) < batch_size:
                break

            files.clear()

        os.rmdir('temp')
        
        utils.create_log('preprocess')

        os.chdir('..')

    msg(f'executed "preprocess sequence"')

def combine(preprocess_dir, data, batch_size, n_processes, max_features, verbose):
    """
    Preprocess a gunzip (gz) compressed text file containing genome sequence data of the following sparse format

        \b
        sequence_1 | identifier_{1,1}:count_{1,1} identifier_{1,1}:count_{2,1} ...
        sequence_2 | identifier_{2,1}:count_{2,1} identifier_{2,1}:count_{2,2} ...
        ...

    and combines the preprocessed data with the `preprocess_dir` directory set in the \033[1mactive config\033[0m file.
    This relies on the user to have previously executed `genolearn preprocess`.

    See https://genolearn.readthedocs.io/tutorial/combine for more details.
    """

    from   genolearn.logger       import msg, Waiting
    from   genolearn              import utils
    from   pathos.multiprocessing import cpu_count, Pool

    import resource

    import json
    import gzip

    assert os.path.exists(preprocess_dir)

    try:
        # remove number of files open restriction
        limit = min(resource.getrlimit(resource.RLIMIT_NOFILE)[1], batch_size + 100)
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
    except:
        msg(f'Attempting to open to many files! Try reducing the number')
        return 

    with gzip.open(os.path.join(preprocess_dir, 'features.txt.gz')) as gz:
        feature_set = gz.read().decode().split()
        
    n_processes    = cpu_count() if n_processes == 'auto' else int(n_processes)

    gather_feature = lambda line : line[:line.index(' ')]
    gather_samples = lambda line : re.findall(r'[\w]+(?=:)', line)
    gather_counts  = lambda line : re.findall(r'(?<=:)[\w]+', line)

    first_run  = True
    features   = []
    C          = 0
    hi         = 0
    unique     = set()

    if data.endswith('.gz'):
        _open  = gzip.GzipFile
        decode = lambda line : line.decode()
    else:
        _open  = open
        decode = lambda line : line

    with _open(os.path.expanduser(data)) as gz:
       
        os.chdir(preprocess_dir)
        
        if 'temp' in os.listdir():
            shutil.rmtree('temp')

        os.mkdir('temp')
        
        files      = {}
        exceptions = set([file.replace('.npz', '') for file in os.listdir('array')])
        while True:
           
            gz.seek(0)

            skip    = False
            skipped = False
            c       = 0
           
            for m, line in enumerate(gz, 1):

                line    = decode(line)

                srrs    = gather_samples(line)
                counts  = gather_counts(line)

                if first_run:
                    features.append(gather_feature(line))
                    hi      = max(hi, *map(int, counts))
                    unique |= set(srrs)

                for SRR, count in zip(srrs, counts):
                    if SRR not in exceptions:
                        if SRR not in files:
                            if skip:
                                skipped = True
                                continue
                            files[SRR] = init(SRR)
                            c         += 1
                            C         += 1
                            skip = c == batch_size
                        add(files[SRR], m - 1, count)

                msg(f'{C:10,d} {m:10,d}', inline = m % verbose)
               
                if m == max_features:
                    break

            for f in files.values():
                f.close()

            if first_run:
                first_run = False

                n         = len(unique)
                d_dtype   = get_dtype(hi)
                c_dtype   = get_dtype(m)
                
                with open('info.json') as f:
                    meta = json.load(f)
                    n    = meta['n'] + n
                    m    = meta['m']
                    hi   = max(meta['max'], hi)

                f = init('meta', None, 'json')
                json.dump({'n' : n, 'm' : m, 'max' : hi}, f)
                f.close()
                
                feature_overlap = np.nonzero(np.isin(features, feature_set, assume_unique = True))[0]                    
               
                def convert(file):
                    txt    = os.path.join('temp', f'{file}.txt')
                    npz    = f'{file}.npz'
                    c, d   = np.loadtxt(txt, dtype = c_dtype).T
                    mask   = np.nonzero(np.isin(c, feature_overlap, assume_unique = True))[0]
                    c, d   = c[mask], d[mask]
                    arr    = np.zeros(m, dtype = d.dtype)
                    arr[c] = d
                    np.savez_compressed(os.path.join('array', npz), arr = arr)
                    os.remove(txt)

            with Waiting('converting', 'converted', 'to arrays'):
                with Pool(n_processes) as pool:
                    pool.map(convert, list(files))

            if not skipped:
                break

            exceptions |= set(files)

            if len(files) < batch_size:
                break

            files.clear()

        os.rmdir('temp')

        utils.create_log('combine')

        os.chdir('..')

    msg(f'executed "preprocess combine"')


def preprocess_meta(output, meta_path, identifier_column, target_column, group_column, train_values, val_values, ptrain):

    import pandas as pd
    import json

    pdir     = 'preprocess'
    file_dir = os.path.join(pdir, 'array')
    files    = [file.replace('.npz', '') for file in os.listdir(file_dir)]
    meta_df  = pd.read_csv(meta_path).applymap(str)
    meta_df[identifier_column] = meta_df[identifier_column].apply(clean_sample)
    
    valid    = set(files)

    meta_df  = meta_df.loc[meta_df[identifier_column].isin(valid)]

    if group_column == 'None':
        group_column                      = 'train_val'
        n                                 = len(meta_df)
        i                                 = int(n * ptrain + 0.5)
        r                                 = np.random.permutation(n)
        values                            = np.array(['train'] * n)
        values[r[i:]]                     = 'val'
        meta_df[group_column]             = values
        
    groups    = sorted(set(meta_df[group_column]))

    meta_json = {'identifiers' : list(meta_df[identifier_column]), 'targets' : sorted(set(meta_df[target_column]))}

    meta_json['search'] = {}
    meta_json['group']  = {}
    for i, row in meta_df.iterrows():
        meta_json['search'][row[identifier_column]] = row[target_column]

    for group in groups:
        meta_json['group'][group] = list(meta_df.loc[meta_df[group_column] == group, identifier_column])

    meta_json['Train'] = []
    meta_json['Val' ]  = []

    for group in groups:
        if group in train_values:
            meta_json['Train'].append(group)
        elif group in val_values:
            meta_json['Val'  ].append(group)
    
    os.makedirs('meta', exist_ok = True)

    with open(os.path.join('meta', output), 'w') as file:
        print(json.dumps(meta_json, indent = 4), file = file)

    print(f'created "{output}" in meta')
