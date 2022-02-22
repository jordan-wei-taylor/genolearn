from   biolearn.utils  import create_log
from   multiprocessing import Pool

import numpy as np
import os

exceptions = 'features.txt', 'log.txt' 
files      = sorted([file for file in os.listdir('testing') if file.endswith('.txt') and file not in exceptions])

def txt2npy(file):
    npy  = f'testing/{file[:file.index(".")]}.npy'
    np.save(npy, np.loadtxt(f'testing/{file}', dtype = np.uint8))

with Pool(16) as pool:
    pool.map(txt2npy, files)

create_log('testing', 'log-txt2npy.txt')