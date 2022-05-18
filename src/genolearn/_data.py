import numpy as np
import os

output_dir = None

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

def set_output_dir(path):
    global output_dir
    output_dir = path

def set_functions(funcs):
    global functions 
    functions = funcs

def set_files(f):
    global files
    files = f

def init_write(file, subpath = 'temp', ext = 'txt', out = None):
    global output_dir
    out  = output_dir if out is None else out
    path = os.path.join(out, subpath, f'{file}.{ext}') if subpath else os.path.join(out, f'{file}.{ext}')
    if os.path.exists(path):
        os.remove(path)
    return open(path, 'a')

def init_dict(file, subpath = 'temp', ext = 'txt'):
    return {}

def add_write(object, i, count):
    object.write(f'{i} {count}\n')

def add_dict(object, i, count):
    object[i] = int(count)

def set_memory(not_low_memory):
    global init, add
    init, add = [[init_write, add_write],
                 [init_dict , add_dict  ]][not_low_memory]