# logging and utility functions
from   biolearn.logger import msg
from   biolearn.utils  import to_txt, to_json, create_log

# parsing meta data TODO: replace with built-in open function
import pandas as pd

import shutil
import gzip
import re
import os

# helper functions for reading sparse data
gather_feature = lambda line : line[:line.index(' ')]
gather_samples = lambda line : re.findall(r'[\w]+(?=:)', line)
gather_counts  = lambda line : re.findall(r'(?<=:)[\w]+', line)
gather         = lambda line : (f(line) for f in (gather_feature, gather_samples, gather_counts))

def check(output_directory):
    """ creates `output_directory` - if it already exists, clean out contents """
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)

def get_meta(meta_path, identifier, *columns):
    """ reads in meta data to DataFrame using `meta_path` and checks identifier and columns are valid column names of the DataFrame """
    meta = pd.read_csv(meta_path, sep = '\t')
    
    if identifier not in meta:
        raise Exception(f'"{identifier}" not a column in meta data csv!')

    meta = meta.set_index(identifier)

    for column in columns:
        if column not in meta:
            raise Exception(f'"{column}" not a column in meta data csv!')
    
    valid_samples = set(meta.index)

    return meta, valid_samples

def process_groupby(input_path, meta_path, identifier, groupby, output_directory, targets):

    check(output_directory)

    meta, valid_samples = get_meta(meta_path, identifier, *targets, groupby)

    groups = set(meta[groupby])

    with gzip.open(input_path) as gz:
        Samples  = set()
        for N, line in enumerate(gz, 1):
            Samples |= set(gather_samples(line.decode()))
            msg(f'read line {N:,d}', inline = True)

        msg(f'read {N:,d} lines')
        
        Groupby = {}
        for group in groups:
            mask    = meta[groupby] == group
            samples = list(set(meta.loc[mask].index) & Samples)
            Groupby[group] = (samples, len(samples))
            
        Sample2Group = {}
        Sample2IDX   = {}
        for (group, (samples, n)) in Groupby.items():
            for i, sample in enumerate(samples):
                Sample2Group[sample] = group
                Sample2IDX[sample]   = i

        X = {}
        for group in groups:
            path = os.path.join(output_directory, str(group))
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)

            X[group] = gzip.open(os.path.join(path, 'X.txt.gz'), 'ab')
            
            meta.loc[Groupby[group][0], targets].to_csv(os.path.join(path, 'Y.txt'), header = True, index = True)

        gz.seek(0)
            
        initials = {group : ['0'] * Groupby[group][1] for group in groups}

        v        = {group : 0 for group in groups}
        features = []
        for j, line in enumerate(gz, 1):
            feature, samples, counts = gather(line.decode())
            copy = {group : value.copy() for group, value in initials.items()}
            for sample, count in zip(samples, counts):
                if sample in valid_samples:
                    copy[Sample2Group[sample]][Sample2IDX[sample]] = count
                    v[Sample2Group[sample]] += 1
            for group in groups:
                X[group].write(f'{" ".join(copy[group])}\n'.encode())
            features.append(feature)
            msg(f'processed {j:,d} of {N:,d} lines', inline = True)
        
        msg(f'processed {N:,d} lines')
        
        for group in groups:
            to_json(dict(n = Groupby[group][1], m = N, v = v[group]), os.path.join(output_directory, str(group), 'meta.json'))
            for file in ['X.txt.gz', 'Y.txt']:
                msg(f'written "{os.path.join(output_directory, str(group), file)}"')

        to_txt(' '.join(features), os.path.join(output_directory, 'features.txt'))
        to_json(Sample2Group, os.path.join(output_directory, 'samples2groups.json'))
        to_json(Sample2IDX, os.path.join(output_directory, 'samples2idx.json'))
        create_log(output_directory)

def process(input_path, meta_path, identifier, output_directory, targets):

    check(output_directory)

    meta, valid_samples = get_meta(meta_path, identifier, *targets)

    with gzip.open(input_path) as gz:
        Samples  = set()
        for j, line in enumerate(gz, 1):
            Samples |= set(gather_samples(line.decode()))
            msg(f'read line {j:,d}', inline = True)

        N = j + 1

        msg(f'read {N:,d} lines')

        Sample2IDX = {sample : i for i, sample in enumerate(Samples & valid_samples)}

        gz.seek(0)

        initial = ['0'] * len(Sample2IDX)

        X        = gzip.open(os.path.join(output_directory, 'X.txt.gz'), 'ab')

        meta.loc[list(Sample2IDX), targets].to_csv(os.path.join(output_directory, 'Y.txt'), header = True, index = True)

        v        = 0
        features = []
        for j, line in enumerate(gz, 1):
            feature, samples, counts = gather(line.decode())
            copy = initial.copy()
            for sample, count in zip(samples, counts):
                if sample in valid_samples:
                    copy[Sample2IDX[sample]] = count
                    v += 1
            X.write(f'{" ".join(copy)}\n'.encode())
            features.append(feature)
            msg(f'processed {j:,d} of {N:,d} lines', inline = True)
        
        msg(f'processed {N:,d} lines')

        for file in ['X.txt.gz', 'Y.txt']:
            msg(f'written "{os.path.join(output_directory, file)}"')

        to_json(dict(n = len(Sample2IDX), m = N, v = v), os.path.join(output_directory, 'meta.json'))

        to_txt(' '.join(features), os.path.join(output_directory, 'features.txt'))
        to_json(Sample2IDX, os.path.join(output_directory, 'samples2idx.json'))
        create_log(output_directory)

# if __name__ == '__main__':

#     from   argparse import ArgumentParser, RawTextHelpFormatter

#     parser = ArgumentParser(description = description, formatter_class = RawTextHelpFormatter)

#     parser.add_argument('input_file')
#     parser.add_argument('meta_path')
#     parser.add_argument('identifier')
#     parser.add_argument('output_directory')
#     parser.add_argument('target', nargs='+')
#     parser.add_argument('--groupby', default = False)

#     args   = parser.parse_args()
#     params = dict(args._get_kwargs())
#     print_dict('executing "process.py" with parameters:', params)

#     start(params)

#     if args.groupby:
#         process_groupby(args.input_file, args.meta_path, args.identifier, args.groupby, args.output_directory, args.target)
#     else:
#         process(args.input_file, args.meta_path, args.identifier, args.output_directory, args.target)
    
#     msg('executed "process.py"')
