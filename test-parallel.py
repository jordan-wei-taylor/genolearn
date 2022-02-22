import gzip
import pandas as pd
import re
import os
import gc
from   multiprocessing import Pool
from   biolearn.logger import Counter
from   biolearn.utils  import create_log

meta           = pd.read_csv('raw-data/meta_data.csv', sep = '\t')

gather_feature = lambda line : line[:line.index(' ')]
gather_samples = lambda line : re.findall(r'[\w]+(?=:)', line)
gather_counts  = lambda line : re.findall(r'(?<=:)[\w]+', line)
gather         = lambda line : (f(line) for f in (gather_feature, gather_samples, gather_counts))

base           = 'testing-parallel'
os.makedirs(base, exist_ok = True)

for file in os.listdir(base):
    os.remove(f'{base}/{file}')

Samples = set()
with gzip.GzipFile('raw-data/STEC_14-19_fsm_kmers.txt.gz') as gz:
    for _ in range(100):
        line = gz.readline()
        feature, samples, counts = gather(line.decode())
        Samples |= set(samples)
        
gather  = lambda line : dict(zip(gather_samples(line), gather_counts(line)))

Samples = list(Samples)
N       = len(Samples)

def clean_open(file):
    path = f'{base}/{file}.txt'
    if os.path.exists(path):
        os.remove(path)
    return open(path, 'a')

with gzip.GzipFile('raw-data/STEC_14-19_fsm_kmers.txt.gz') as gz:

    files = {name : clean_open(name) for name in Samples}
    counter = Counter('processing feature {:11,d}', 1000000)

    def process(i, line):
        counter.increment()
        line    = line.decode()
        feature = gather_feature(line)

        srr_count = gather(line)

        for SRR in files:
            if SRR in srr_count:
                files[SRR].write(f'{i} {srr_count[SRR]}\n')

        del srr_count, SRR; gc.collect()

        return feature

    with Pool(8) as pool:
        features = pool.starmap(process, enumerate(gz))
    
    for f in files.values():
        f.close()

f = clean_open('features')
f.write(' '.join(features))
f.close()

create_log(base)
