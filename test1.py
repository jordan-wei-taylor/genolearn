def main(output_dir, genome_sequence, batch_size, verbose, n_processes):

    if batch_size == -1:
        batch_size = np.inf

    n_processes    = cpu_count() if n_processes == 'auto' else int(n_processes)

    gather_feature = lambda line : line[:line.index(' ')]
    gather_samples = lambda line : re.findall(r'[\w]+(?=:)', line)
    gather_counts  = lambda line : re.findall(r'(?<=:)[\w]+', line)
    gather         = lambda line : dict(zip(gather_samples(line), gather_counts(line)))

    

    def clean_open(file):
        path = f'{output_dir}/{file}.txt'
        if os.path.exists(path):
            os.remove(path)
        return open(path, 'a')

    os.makedirs(output_dir, exist_ok = True)

    for file in os.listdir(output_dir):
        os.remove(f'{output_dir}/{file}')

    first_run  = True
    features   = []
    exceptions = set()
    C          = 0
        
    with gzip.GzipFile(genome_sequence) as gz:
        
        while True:
            
            gz.seek(0)

            skip  = False
            c     = 0
            files = {}

            for i, line in enumerate(gz):

                line = line.decode()

                if first_run:
                    features.append(gather_feature(line))

                srr_count = gather(line)

                for SRR, count in srr_count.items():
                    if SRR not in exceptions:
                        if skip:
                            continue
                        elif SRR not in files:
                            files[SRR] = clean_open(SRR)
                            c         += 1
                            C         += 1
                            skip = c == batch_size
                        files[SRR].write(f'{i} {srr_count[SRR]}\n')

                if i % verbose == 0:
                    msg(f'{C:10,d} {i:10,d}')
            
            msg(f'{C:10,d} {i + 1:10,d}')

            for f in files.values():
                f.close()
            
            with Pool(n_processes) as pool:
                pool.map(txt2npy, list(files))

            if first_run:
                f = clean_open('features')
                f.write(' '.join(features))
                f.close()
                features.clear()
                first_run = False

            exceptions |= set(files)

            if len(files) < batch_size:
                break

            files.clear()


        create_log(output_dir)

if __name__ == '__main__':

    from   biolearn.logger  import print_dict

    from   argparse import ArgumentParser, RawTextHelpFormatter

    import gzip
    import numpy  as np
    import re
    import os

    from   multiprocessing import cpu_count, Pool

    from   biolearn.logger import msg
    from   biolearn.utils  import create_log

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

    args   = parser.parse_args()
    params = dict(args._get_kwargs())
    print_dict('executing "process.py" with parameters:', params)

    def txt2npy(file):
        txt = f'{args.output_dir}/{file}.txt'
        npy = f'{args.output_dir}/{file}.npy'
        np.save(npy, np.loadtxt(txt, dtype = np.uint8))
        os.remove(txt)

    main(args.output_dir, args.genome_sequence_path, args.batch_size, args.verbose, args.n_processes)