if __name__ == '__main__':

    from   genolearn.logger  import print_dict, msg
    from   genolearn         import utils, _data

    from   argparse          import ArgumentParser, RawTextHelpFormatter
    from   multiprocessing   import cpu_count, Pool
    from   shutil            import rmtree

    import numpy  as np

    import json
    import gzip
    import re
    import os

    description = \
    r"""
    Processes a gunzip (gz) compressed text file containing genome sequence data of the following sparse format

        sequence_1 | identifier_{1,1}:count_{1,1} identifier_{1,1}:count_{2,1} ...
        sequence_2 | identifier_{2,1}:count_{2,1} identifier_{2,1}:count_{2,2} ...
        ...

    into a directory of .npz files, a list of all the features, and some meta information containing number of
    identifiers, sequences, and non-zero counts.

    Required Arguments
    =======================
        output_dir           : output directory
        genome_sequence_path : path to compressed text file with sparse format

    Optional Arguments
    =======================
        batch_size  = 512    : number of temporary txt files to generate over a single parse of the genome data
        verbose     = 250000 : number of iterations before giving verbose update
        n_processes = 'auto' : number of processes to run in parallel when compressing txt to npy files
        sparse      = True   : output sparse npz files
        dense       = True   : output dense npz files
        debug       = -1     : integer denoting first number of features to consider (-1 results in all features)

    Optional Flags
    =======================
        --not_low_memory     : if not flagged, will write to temporary txt files before converting to npz files, otherwise, will consume RAM to then generate the npz files

    Example Usage
    =======================
        python -m genolearn data raw-data/STEC_14-19_fsm_kmers.txt.gz --batch_size 256
    """

    parser = ArgumentParser(description = description, formatter_class = RawTextHelpFormatter)

    parser.add_argument('output_dir')
    parser.add_argument('genome_sequence_path')
    parser.add_argument('-batch_size', type = int, default = 512)
    parser.add_argument('-verbose', type = int, default = 250000)
    parser.add_argument('-n_processes', default = 'auto')
    parser.add_argument('-sparse', default = True, type = bool)
    parser.add_argument('-dense', default = True, type = bool)
    parser.add_argument('-debug', default = -1, type = int)
    parser.add_argument('--not_low_memory', default = False, action = 'store_true')

    args   = parser.parse_args()
    params = dict(args._get_kwargs())
    print_dict('executing "genolearn" with parameters:', params)

    if args.batch_size == -1:
        args.batch_size = np.inf

    args.n_processes = cpu_count() if args.n_processes == 'auto' else int(args.n_processes)

    gather_feature = lambda line : line[:line.index(' ')]
    gather_samples = lambda line : re.findall(r'[\w]+(?=:)', line)
    gather_counts  = lambda line : re.findall(r'(?<=:)[\w]+', line)

    _data.set_memory(args.not_low_memory)
    _data.set_output_dir(args.output_dir)
    
    if os.path.exists(args.output_dir):
        rmtree(args.output_dir)

    os.makedirs(f'{args.output_dir}/temp', exist_ok = True)
    os.makedirs(os.path.join(args.output_dir, 'feature-selection'), exist_ok = True)

    first_run  = True
    features   = []
    exceptions = set()
    C          = 0
    hi         = 0
    unique     = set()

    with gzip.GzipFile(args.genome_sequence_path) as gz:
        
        files = {}
        _data.set_files(files)

        while True:
            
            gz.seek(0)

            skip    = False
            skipped = False
            c       = 0
            
            for m, line in enumerate(gz, 1):

                line   = line.decode()

                srrs   = gather_samples(line)
                counts = gather_counts(line)

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
                            files[SRR] = _data.init(SRR)
                            c         += 1
                            C         += 1
                            skip = c == args.batch_size
                        _data.add(files[SRR], m - 1, count)

                if m % args.verbose == 0:
                    msg(f'{C:10,d} {m:10,d}')
                
                if m == args.debug:
                    break
            
            if m % args.verbose:
                msg(f'{C:10,d} {m:10,d}')

            if not args.not_low_memory:
                for f in files.values():
                    f.close()

            if first_run:
                first_run = False

                n         = len(unique)
                d_dtype   = utils.get_dtype(hi)
                c_dtype   = utils.get_dtype(m)
                r_dtype   = utils.get_dtype(n)

                utils.set_m(m)
                
                gzf = _data.init_write('features', None, 'txt.gz', args.output_dir)
                gzf.write(gzip.compress(' '.join(features).encode()))
                gzf.close()
                features.clear()

                f = _data.init_write('meta', None, 'json', args.output_dir)
                json.dump({'n' : len(unique), 'm' : m, 'max' : hi}, f)
                f.close()
                
                def to_sparse(npz, c, d):
                    np.savez_compressed(os.path.join(args.output_dir, 'sparse', npz), col = c.astype(c_dtype), data = d.astype(d_dtype))

                def to_dense(npz, c, d):
                    arr = np.zeros(m, dtype = d.dtype)
                    arr[c] = d
                    np.savez_compressed(os.path.join(args.output_dir, 'dense', npz), arr = arr)
                
                def convert_write(file):
                    txt  = os.path.join(args.output_dir, 'temp', f'{file}.txt')
                    npz  = f'{file}.npz'
                    c, d = np.loadtxt(txt, dtype = c_dtype).T

                    for function in functions:
                        function(npz, c, d)

                    os.remove(txt)

                def convert_dict(file):
                    npz  = f'{file}.npz'
                    c, d = map(np.array, map(list, zip(*files[file].items())))

                    for function in functions:
                        function(npz, c, d)

                convert = convert_dict if args.not_low_memory else convert_write

                functions = []
                if args.sparse:
                    functions.append(to_sparse)
                    os.makedirs(f'{args.output_dir}/sparse')

                if args.dense:
                    functions.append(to_dense)
                    os.makedirs(f'{args.output_dir}/dense')

                _data.set_functions(functions)
            
            with Pool(args.n_processes) as pool:
                pool.map(convert, list(files))

            if not skipped:
                break

            exceptions |= set(files)

            if len(files) < args.batch_size:
                break

            files.clear()

        os.rmdir(os.path.join(args.output_dir, 'temp'))

        utils.create_log(args.output_dir)
    
    msg('executed "genolearn"')
