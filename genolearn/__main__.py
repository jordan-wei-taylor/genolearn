from re import M


if __name__ == '__main__':

    from   genolearn.logger  import print_dict, msg
    from   genolearn         import utils

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

    feature_id_1 | sample_id_1:value_1_1 sample_id_2:value_1_2 ...\n
    feature_id_2 | ...

    into a gunzip compressed text file which contains a matrix. The ij-th element of the matrix refers to the value at the 
    i-th feature and j-th sample i.e. value_i_j at feature_id_i, sample_id_j.

    Required Arguments
    =======================
        output_dir           : output directory
        genome_sequence_path : path to compressed text file with sparse format

    Optional Arguments
    =======================
        batch_size  = 512    : number of temporary txt files to generate over a single parse of the genome data
        verbose     = 250000 : number of iterations before giving verbose update
        n_processes = 'auto' : number of processes to run in parallel when compressing txt to npy files

    Example Usage
    =======================
        python -m biolearn data raw-data/STEC_14-19_fsm_kmers.txt.gz --batch_size 256
    """

    parser = ArgumentParser(description = description, formatter_class = RawTextHelpFormatter)

    parser.add_argument('output_dir')
    parser.add_argument('genome_sequence_path')
    parser.add_argument('-batch_size', type = int, default = 512)
    parser.add_argument('-verbose', type = int, default = 250000)
    parser.add_argument('-n_processes', default = 'auto')
    parser.add_argument('-sparse', default = True, type = bool)
    parser.add_argument('-dense', default = True, type = bool)
    parser.add_argument('-low_memory', default = True, type = bool)

    args   = parser.parse_args()
    params = dict(args._get_kwargs())
    print_dict('executing "genolearn" with parameters:', params)

    if args.batch_size == -1:
        args.batch_size = np.inf

    args.n_processes = cpu_count() if args.n_processes == 'auto' else int(args.n_processes)

    gather_feature = lambda line : line[:line.index(' ')]
    gather_samples = lambda line : re.findall(r'[\w]+(?=:)', line)
    gather_counts  = lambda line : re.findall(r'(?<=:)[\w]+', line)

    if os.path.exists(args.output_dir):
        rmtree(args.output_dir)

    os.makedirs(f'{args.output_dir}/process/', exist_ok = True)

    def clean_open(file, subpath = 'process', ext = 'txt'):
        path = os.path.join(args.output_dir, subpath, f'{file}.{ext}') if subpath else os.path.join(args.output_dir, f'{file}.{ext}')
        if os.path.exists(path):
            os.remove(path)
        return open(path, 'a')

    def get_dtype(val):
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
        for dtype in dtypes:
            info = np.iinfo(dtype)
            if info.min <= val <= info.max:
                return dtype
        raise Exception()

    os.makedirs(os.path.join(args.output_dir, 'feature-selection'), exist_ok = True)

    first_run  = True
    features   = []
    exceptions = set()
    C          = 0
    hi         = 0
    unique     = set()

    with gzip.GzipFile(args.genome_sequence_path) as gz:
        
        while True:
            
            gz.seek(0)

            skip    = False
            skipped = False
            c       = 0
            files   = {}
            
            for i, line in enumerate(gz):

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
                            files[SRR] = clean_open(SRR)
                            c         += 1
                            C         += 1
                            skip = c == args.batch_size
                        files[SRR].write(f'{i} {count}\n')

                if i % args.verbose == 0:
                    msg(f'{C:10,d} {i:10,d}')

            msg(f'{C:10,d} {i + 1:10,d}')

            for f in files.values():
                f.close()

            if first_run:
                first_run = False

                n         = len(unique)
                m         = i + 1
                d_dtype   = get_dtype(hi)
                c_dtype   = get_dtype(m)
                r_dtype   = get_dtype(n)

                utils.set_m(m)
                utils.set_d_dtype(d_dtype)
                utils.set_c_dtype(c_dtype)
                utils.set_r_dtype(r_dtype)
                
                f = clean_open('features', None)
                f.write(' '.join(features))
                f.close()
                features.clear()

                f = clean_open('meta', None, 'json')
                json.dump({'n' : len(unique), 'm' : m, 'max' : hi}, f)
                f.close()

                functions = []
                if args.sparse:
                    functions.append(utils.process2sparse)
                    os.makedirs(f'{args.output_dir}/sparse')

                if args.dense:
                    functions.append(utils.process2dense)
                    os.makedirs(f'{args.output_dir}/dense')

                def convert(file):
                    txt  = f'{args.output_dir}/process/{file}.txt'
                    npz  = f'{file}.npz'
                    c, d = np.loadtxt(txt, dtype = c_dtype).T

                    for function in functions:
                        function(args.output_dir, npz, c, d)

                    os.remove(txt)

            with Pool(args.n_processes) as pool:
                pool.map(convert, list(files))

            if not skipped:
                break

            exceptions |= set(files)

            if len(files) < args.batch_size:
                break

            files.clear()

        os.rmdir(f'{args.output_dir}/process')

        utils.create_log(args.output_dir)
    
    msg('executed "genolearn"')
