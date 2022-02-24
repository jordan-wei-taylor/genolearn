import gzip
import pandas as pd
import re
import os

from   biolearn.logger import msg
from   biolearn.utils  import create_log

meta           = pd.read_csv('raw-data/meta_data.csv', sep = '\t')

gather_feature = lambda line : line[:line.index(' ')]
gather_samples = lambda line : re.findall(r'[\w]+(?=:)', line)
gather_counts  = lambda line : re.findall(r'(?<=:)[\w]+', line)
gather         = lambda line : dict(zip(gather_samples(line), gather_counts(line)))

os.makedirs('testing', exist_ok = True)

for file in os.listdir('testing'):
    os.remove(f'testing/{file}')

features = []

def clean_open(file):
    path = f'testing/{file}.txt'
    if os.path.exists(path):
        os.remove(path)
    return open(path, 'a')

with gzip.GzipFile('raw-data/STEC_14-19_fsm_kmers.txt.gz') as gz:

    files = {}

    for i, line in enumerate(gz):

        line = line.decode()

        features.append(gather_feature(line))

        srr_count = gather(line)

        for SRR, count in srr_count.items():
            if SRR not in files:
                files[SRR] = clean_open(SRR)
            files[SRR].write(f'{i} {srr_count[SRR]}\n')

        if i % 1000000 == 0:
            msg(f'{i:10,d}')
    
    for f in files.values():
        f.close()

f = clean_open('features')
f.write(' '.join(features))
f.close()

create_log('testing')
