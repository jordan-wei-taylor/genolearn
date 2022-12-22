from   genolearn.logger import msg

from   time             import time
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

def _gen_prompt(msg, type, default ,space, default_option, multiple, show_choices):
    if multiple:
        msg = f'{msg}*'
    msg = f'{msg}' + ' ' * space
    if isinstance(type, click.Choice) and show_choices:
        msg = f'{msg} ' + '{' + ', '.join(map(str, type.choices)) + '}'
    if default_option:
       msg = f'{msg} [{default}]' 
    return msg

def _check(value, type):
    if value == 'None':
        value = None
    try:
        type(value)
        return True
    except:
        return False

def _warn(value, type):
    print(f'bad parameter {value} not {type}')

def _convert_range(string):
    nums   = re.findall('(?<=range\()[0-9, ]+', string)[0].replace(',', ' ').split()
    ls     = list(map(str, list(range(*map(int, nums)))))
    string = ','.join(ls)
    return string

def _prompt(msg, type, default, default_option = True, multiple = False, show_choices = False):

    # user input
    value = input(msg + ': ')

    # check default
    if value == '':
        if default_option:
            value = str(default)
        else:
            print('user input required!')
            return _prompt(msg, type, default, default_option, multiple, show_choices)

    # check range
    if 'range' in value:

        # warn and re-input if type is not int
        if (type != click.INT) and not isinstance(type, click.IntRange):
            _warn(value, type)
            return _prompt(msg, type, default, default_option, multiple, show_choices)

        # convert range string to list of integers
        value = _convert_range(value)

    if value == '*':
        value = ','.join(type.choices)

    # detect if multi-value input
    if ',' in value:
        values = re.split(' ?, ?', value)

        # check each value and return if correct inputs
        for value in values:
            check = _check(value, type)
            if not check:
                _warn(value, type)
                return _prompt(msg, type, default, default_option, multiple, show_choices)
        values = [None if value == None else value for value in map(type, values)]
        return values

    # check (single) value and return if correct input
    check = _check(value, type)
    if check:
        if value == 'None':
            value = None
            return value
        return [type(value)] if multiple else type(value)
    _warn(value, type)
    return _prompt(msg, type, default, default_option, multiple, show_choices)
    
def _default(info):
    if isinstance(info['type'], click.Choice) and len(info['type'].choices) == 1:
        return info['type'].choices[0]
    else:
        return info.get('default', None)

def prompt(params):
    prompts = []
    for key, info in params.items():
        default = _default(info)
        show    = info.get('show_choices', False)
        prompt = _gen_prompt(info.get('prompt', key), info['type'], default, 0, 'default' in info or default, info.get('multiple', False), show)
        prompts.append(prompt)
    
    length = max(map(len, prompts))
    config = {}
    for i, (key, info) in enumerate(params.items()):
        if isinstance(info['type'], click.Choice) and len(info['type'].choices) == 1:
            config[key] = info['type'].choices[0]
            continue
        default = _default(info)
        flag    = 'default' in info or default
        value   = _prompt(f'{prompts[i]:{length}s}', info['type'], default, flag, info.get('multiple', False), info.get('show_choices', False))
        config[key] = value
    return config

START    = time()
RAMSTART = get_process_memory()
RAM      = RAMSTART
DATETIME = datetime.now().strftime('%d-%m-%Y %X')