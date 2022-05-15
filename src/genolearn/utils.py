from   genolearn.logger import msg

from   time           import time
from   datetime       import datetime

import psutil
import json
import os

import numpy as np

PARAMS = {}

def get_process_memory():
    """ returns current RAM usage by current Python process """
    return psutil.Process(os.getpid()).memory_info().rss

def monitor_RAM():
    global RAM
    RAM = max(RAM, get_process_memory())
    
def set_params(params):
    """ sets the global PARAMS value """
    global PARAMS
    PARAMS = params

def create_log(path = '.', filename = 'log.txt'):
    """ creates a text log file with total time elapsed and RAM usage """
    global DATETIME, START, RAM, RAMSTART, PARAMS
    dur  = time() - START
    monitor_RAM()
    ram  = (RAM - RAMSTART) / 1024 ** 3 # B -> GB conversion
    m, s = divmod(dur, 60)
    h, m = divmod(m  , 60)
    contents = f'datetime : {DATETIME}\nduration : {h:02.0f}h {m:02.0f}m {s:02.0f}s\nmemory   : {ram:.3f} GB'

    if PARAMS:
        contents = f'{contents}\n\n{json.dumps(PARAMS, indent = 4)}'

    to_txt(contents, os.path.join(path, filename))

def to_txt(obj, path, mode = 'w'):
    """ writes an object to a text file """
    msg(f'writing "{path}"', inline = True)
    with open(path, mode) as f:
        f.write(obj)
    msg(f'written "{path}"')

def to_json(obj, path):
    """ writes an object to a json file """
    msg(f'writing "{path}"', inline = True)
    with open(path, 'w') as f:
        json.dump(obj, f)
    msg(f'written "{path.split(os.path.sep)[-1] if os.path.sep in path else path}"')

def get_dtype(val):
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtypes:
        info = np.iinfo(dtype)
        if info.min <= val <= info.max:
            return dtype
    raise Exception()
    
def set_c_dtype(dtype):
    global c_dtype
    c_dtype = dtype

def set_r_dtype(dtype):
    global r_dtype
    r_dtype = dtype

def set_d_dtype(dtype):
    global d_dtype
    d_dtype = dtype

def set_m(val):
    global m
    m = val

def process2sparse(npz, c, d):
    np.savez_compressed(npz, col = c, data = d.astype(d_dtype))

def process2dense(npz, c, d):
    arr = np.zeros(m, dtype = d.dtype)
    arr[c] = d
    np.savez_compressed(npz, arr = arr)

def subdir(path, sub, ext = 0):
    _sub = f'{sub}-{ext}' if ext else sub
    if _sub in os.listdir(path):
        return subdir(path, sub, ext + 1)
    return _sub
    
def check_config(config):
    import json
    import re

    if config:
        if '{' in config:
            raw    = config
            while 'range' in raw:
                nums = re.findall('(?<=range\()[0-9, ]+', raw)[0].replace(',', ' ').split()
                raw  = re.sub('range\([0-9, ]+\)', str(list(range(*map(int, nums)))), raw, 1)
            config = json.loads(raw)
        else:
            with open(config) as f:
                config = json.load(f)
                if isinstance(config, str):
                    config = json.loads(config)
    else:
        config = {}

    return config

def generate_config(path, **kwargs):
    config = {}
    for key, val in kwargs.items():
        if isinstance(val, range):
            val = list(val)
        config[key] = val
    with open(path, 'w') as f:
        json.dump(json.dumps(config, indent = 4), f)

START    = time()
RAMSTART = get_process_memory()
RAM      = RAMSTART
DATETIME = datetime.now().strftime('%d-%m-%Y %X')