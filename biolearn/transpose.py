if __name__ == '__main__':
    from   biolearn.logger import msg, print_dict, Computing
    from   biolearn.utils  import create_log
    import gzip
    import os

    import argparse

    parser = argparse.ArgumentParser(description = 'transposes a gz file')
    parser.add_argument('path')
    parser.add_argument('--batch_size', default = 64, type = int)

    args = parser.parse_args()
    
    params = dict(args._get_kwargs())
    print_dict('executing "transpose.py" with parameters:', params)

    in_file  = os.path.join(args.path, 'X.txt.gz')
    out_file = os.path.join(args.path, 'X-t.txt.gz')

    if os.path.exists(out_file):
        os.remove(out_file)

    with gzip.open(in_file) as gz_in, gzip.open(out_file, 'ab') as gz_out:
        N = gz_in.readline().decode().count(' ') + 1
        gz_in.seek(0)
        start = 0
        end   = min(args.batch_size, N)
        batch = []
        while start < N:
            
            batch.clear()
            
            with Computing(f'batches {start + 1:,d} to {end:,d} of {N:,d}', inline = True):

                for line in gz_in:
                    batch.append(line.decode().rstrip().split()[start:end])

                lines = '\n'.join(' '.join(row) for row in zip(*batch)) + '\n'

                gz_out.write(lines.encode())

                start = end
                end   = min(start + args.batch_size, N)

                gz_in.seek(0)

    create_log(args.path, 'transpose-log.txt')
    
    msg('executed "transpose.py"')