from   genolearn.logger import msg, clear, up

from   time             import time, sleep
from   datetime         import datetime

import psutil
import click
import json
import os
import re


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
    PARAMS = params.copy()

def get_params():
    """ gets the global PARAMS value """
    global PARAMS
    return PARAMS.copy()
    
def create_log(filename, path = '.'):
    """ creates a text log file with total time elapsed and RAM usage """
    global DATETIME, START, RAM, RAMSTART, PARAMS
    dur  = time() - START
    monitor_RAM()
    ram  = (RAM - RAMSTART) / 1024 ** 3 # B -> GB conversion
    m, s = divmod(dur, 60)
    h, m = divmod(m  , 60)
    contents = f'datetime : {DATETIME}\nduration : {h:02.0f}h {m:02.0f}m {s:02.0f}s\nmemory   : {ram:.3f} GB'

    if PARAMS:
        for key, value in PARAMS.items():
            if isinstance(value, str):
                PARAMS[key] = value.replace(os.path.expanduser('~'), '~')
        contents = f'{contents}\n\n{json.dumps(PARAMS, indent = 4)}'

    contents = f'{contents}\n'
    
    if not filename.endswith('.log'):
        filename = f'{filename}.log'
        
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

def subdir(path, sub, ext = 0):
    _sub = f'{sub}-{ext}' if ext else sub
    if _sub in os.listdir(path):
        return subdir(path, sub, ext + 1)
    return _sub

def _gen_prompt(key, info, space = 0):
    msg = info.get('prompt', key)
    if info.get('multiple', False):
        msg = f'{msg}*'
    msg = f'{msg}' + ' ' * space
    if isinstance(info['type'], click.Choice) and info.get('show_choices', False):
        msg = f'{msg} ' + '{' + ', '.join(map(str, info['type'].choices)) + '}'
    if 'default' in info:
       msg = f'{msg} [{info["default"]}]' 
    return msg

def _check(value, type):
    if value == 'None':
        value = None
    try:
        type(value)
        return True
    except:
        return False

def pause(func):
    def wrapper(*args):
        print(up, end = '')
        # needs repitition as sleep may execute before func(*args) resulting in warning to not show
        for _ in range(4):
            func(*args)
            sleep(1)
        print(clear, end = '')
    return wrapper

@pause
def _warn_missing(msg):
    print(f'\r{msg}: \033[93muser input required!\033[0m', end = '')

@pause  
def _warn(msg, value, dtype):
    print(f'\r{msg}: {value}  \033[93mbad parameter {value} not {dtype}\033[0m', end = '')
        
def _convert_range(string):
    nums   = re.findall('(?<=range\()[0-9, ]+', string)[0].replace(',', ' ').split()
    ls     = list(map(str, list(range(*map(int, nums)))))
    string = ','.join(ls)
    return string

def _prompt(key, info, space = 0):

    msg   = _gen_prompt(key, info, space)
        
    # user input
    value = input(f'{msg}: ')

    # convenience
    dtype = info['type']

    # check default
    if value == '':
        if 'default' in info:
            value = str(_default(info))
        else:
            _warn_missing(msg)
            return _prompt(key, info, space)

    # utility to take the remainder of the choices if value = *
    if value == '*' and isinstance(dtype, click.Choice):
        value = ','.join(dtype.choices)

    # utility to take default value as * e.g. *-example -> {default}-example
    if value.startswith('*'):
        if value.count('*') == 1 and 'default' in info:
            value = value.replace('*', info['default'])
        else:
            # no default setting
            _warn(msg, value, 'valid')
            return _prompt(key, info, space)

    # check range
    if 'range' in value:

        # warn and re-input if type is not int
        if (dtype != click.INT) and not isinstance(dtype, click.IntRange):
            _warn(msg, value, dtype)
            return _prompt(key, info, space)

        # convert range string to list of integers
        value = _convert_range(value)

    # detect if multi-value input
    if info.get('multiple'):
        if ',' in value:
            values = re.split(' ?, ?', value)

            # check each value and return if correct inputs
            for value in values:
                check = _check(value, dtype)
                if not check:
                    _warn(msg, value, dtype)
                    return _prompt(key, info, space)
            values = [None if value == None else value for value in map(dtype, values)]
            return values
    elif ',' in value:
        _warn(msg, value, dtype)

    # check (single) value and return if correct input
    check = _check(value, dtype)
    if check:
        if value == 'None':
            value = None
            return value
        return [dtype(value)] if info.get('multiple') else dtype(value)
    _warn(msg, value, dtype)
    return _prompt(key, info, space)
    
def _default(info):
    dtype = info['type']
    if isinstance(info['type'], click.Choice) and len(info['type'].choices) == 1:
        return info['type'].choices[0]
    else:
        return info.get('default')

def prompt(params):
    lengths = []
    for key, info in params.items():
        info.setdefault('type', click.STRING)
        lengths.append(len(_gen_prompt(key, info)))
    length  = max(lengths)
    config  = {}
    for i, (key, info) in enumerate(params.items()):
        space       = length - lengths[i]
        value       = _prompt(key, info, space)
        config[key] = value
    return config

ls = os.listdir()
wd = True if '.genolearn' in ls else False

path = os.path.join(os.path.dirname(__file__), 'wd')
if wd:
    working_directory = os.path.abspath('.')
    with open(path, 'w') as f:
        f.write(working_directory)
elif os.path.exists(path):
    with open(path) as f:
        working_directory = f.read()
        if not (os.path.exists(working_directory) and '.genolearn' in os.listdir(working_directory)):
            working_directory = None
else:
    working_directory = None


ls = os.listdir(working_directory if working_directory and os.path.exists(working_directory) else '.')

class Path():

    def add(self, func):
        name = func.__name__
        if name.startswith('_'):
            name = name[1:]
        self.__dict__[name] = func

path = Path()

def add(func):
    path.add(func)
    return lambda *args, **kwargs : func(*args, **kwargs)
    
@add
def join(*args):
    return os.path.join(working_directory, *args)

@add
def join_from(path):
    return lambda *args : os.path.join(path, *args)

@add
def listdir(path = '.', *args):
    if working_directory and os.path.exists(working_directory):
        path = join(path, *args)
        return os.listdir(path) if os.path.exists(path) else []
    return []

@add
def expanduser(path, inverse = False):
    return path.replace('~', os.path.expanduser('~')) if inverse else path.replace(os.path.expanduser('~'), '~')   

@add
def _open(path, mode = 'r'):
    return open(join(path), mode)

@add
def exists(path):
    return os.path.exists(os.path.expanduser(path))

def get_active():
    try:
        if working_directory:
            path = join('.genolearn')
            if os.path.exists(path):
                import json
                with open(path) as f:
                    return json.load(f)
    except:
        ...


START    = time()
RAMSTART = get_process_memory()
RAM      = RAMSTART
DATETIME = datetime.now().strftime('%d-%m-%Y %X')