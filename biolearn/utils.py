from   biolearn.logger import msg

from   time           import time
from   datetime       import datetime

import psutil
import json
import os

PARAMS = {}

def get_process_memory():
    """ returns current RAM usage by current Python process """
    return psutil.Process(os.getpid()).memory_info().rss

def set_params(params):
    """ sets the global PARAMS value """
    global PARAMS
    PARAMS = params

def create_log(path = '.', filename = 'log.txt'):
    """ creates a text log file with total time elapsed and RAM usage """
    global DATETIME, START, RAM, PARAMS
    dur  = time() - START
    ram  = (get_process_memory() - RAM) / 1024 ** 3 # B -> GB conversion
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
    msg(f'written "{path}"')

START    = time()
RAM      = get_process_memory()
DATETIME = datetime.now().strftime('%d-%m-%Y %X')