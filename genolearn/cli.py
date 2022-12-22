from   genolearn.logger import up, clear, print_dict
from   genolearn.utils  import prompt, _prompt, set_params, get_params
from   genolearn        import __version__, ls, working_directory, get_active, listdir

from   shutil           import rmtree

import resource
import inspect
import pandas as pd
import click
import json
import os


""" general utility variables and functions """

# set preprocessed metadata list
metas  = listdir('meta')

# get active working directory and input metadata
active = get_active()

def user_input(text, n):
    """
    Obtains user input and is bounded between [0, n]
    """
    j = input(text)
    if j.isdigit():
        if 0 <= int(j) < n:
            return int(j), False
    return None, True

def append(command):
    with open(os.path.join(working_directory, '.genolearn')) as file:
        log = json.load(file)
        log['history'].append(command)
    with open(os.path.join(working_directory, '.genolearn'), 'w') as file:
        print(json.dumps(log, indent = 4), file = file, end = '')

def check_working_directory():
    if working_directory:
        if os.path.exists(working_directory):
            if '.genolearn' in listdir():
                return True
    return False

PRE = \
f"""
Genolearn ({__version__}) Command Line Interface

GenoLearn is designed to enable researchers to perform Machine Learning on their genome
sequence data such as fsm-lite or unitig files.

See https://genolearn.readthedocs.io for documentation.
""".strip()

if check_working_directory():
    PRE = f'{PRE}\n\nWorking directory: {working_directory.replace(os.path.expanduser("~"), "~")}'

def enum(options, command = '', pre = '', post = 'user input', k = None, back = None):
    """
    Prints enumerated options to for user input, records the user input, then executes an associated function or returns the option key.

    Parameters
    ----------
        options, (list, dict)
            Available options from the user to choose from. If type dict, the values should be a dictionary with optional entries:
                prompt : prompt to display to the user
                info   : information explaining what the prompt is for
                func   : function to execute should the user choose the option
        
        command, str ['']
            Command text to print prior to pre-text.

        pre, str ['']
            Pre-text to print prior to the enumerated options.

        post, str ['user input']
            Text to print post enumerated options.

        k, int [None]
            Truncate enumerated options.

        back, function [None]
            Function to execute if the user selects to go back (None results in main menu).
    """

    # set back to menu if not provided
    if back is None:
        back = menu

    # set options to only contain back initially (to ensure back is the option 0)
    if back == exit:
        _options = dict(exit = dict(info = 'exits GenoLearn', func = back))
    else:
        _options = dict(back = dict(info = 'goes to the previous command', func = back))

    # add in entries provided
    if isinstance(options, list):
        _options.update({option : {} for i, option in enumerate(options) if k is None or i < k})
    else:
        _options.update({key : value for i, (key, value) in enumerate(options.items()) if k is None or i < k})
        
    options = _options

    # number of options (ignoring back)
    n = len(options)

    # spacing for option number
    m = len(f'{n}')

    # spacing for prompt
    c = max(max(len(option.get('prompt', name)) for name, option in options.items()) + 15, 30)

    # print the pre-text
    if command:
        pre = f'{PRE}\n\nCommand: {command}\n\n{pre}'
    else:
        pre = f'{PRE}\n\nSelect a command'
    print(pre, '\n')

    # print enumerated options
    for i, (name, option) in enumerate(options.items()):
        print(f'{i:{m}d}.  {option.get("prompt", name):{c}s}  {option.get("info", "")}')

        # print extra space to seperate back option from the rest of the option
        if i == 0: print()

    # line space between last option and post-text
    print()

    # obtain user input
    j, flag = user_input(f'{post} : ', n)

    # if incorrect, retry
    while flag:
        print(f'{up}{clear}', end = '')
        j, flag = user_input(f'\r{post} : ', n)

    # clear screen
    pre_n = pre.count('\n') + 1
    print(up * (n + pre_n + 4) + clear, end = '')

    # option key
    key = list(options)[j]

    # if func was provided, execute it
    if 'func' in options[key]:
        func = options[key]['func']
        return func(key) if inspect.signature(func).parameters else func() # if takes in an argument, pass the option key

    # if func not provided, return the option key
    return key

def exit():
    """ Exits the Python shell execution """
    quit()

def _reduce(d, limit):
    """ Truncates dictionary values for printing purposes """
    if isinstance(d, dict):
        if len(d) > limit:
            for i, (k, v) in enumerate(d.copy().items(), 1):
                if i < limit:
                    d[k] = _reduce(v, limit)
                else:
                    del d[k]
                    d['...'] = '...'
        else:
            for k, v in d.items():
                d[k] = _reduce(v, limit)
    elif isinstance(d, list):
        if len(d) > limit:
            d = d[:limit - 1] + ['...']
    return d

def reduce(d, limit):
    """ Truncates dictionary values for printing purposes """
    for k, v in d.items():
        d[k] = _reduce(v, limit)
    return d

def read_log(file):
    """ Reads only the dictionary component of a text file """
    with open(os.path.join(working_directory, file)) as f:
        string = f.read()
        log    = json.loads(string[string.index('{'):])
        return log

def check_history(string):
    """ Checks if any previously executed commands starts with {string}"""
    with open(os.path.join(working_directory, '.genolearn')) as f:
        history = json.load(f)['history']
        for line in history:
            if line.startswith(string):
                return True
        return False

def select_train_dir(command, func, back, train_dir_func = None):
    """ Wrapper for first selecting training directory """
    def _select_train_dir():
        train_dirs = os.listdir(os.path.join(working_directory, 'train'))
        common  = lambda train_dir : {'func' : func, 'info' : train_dir_func(train_dir)} if train_dir_func else {'func' : func}
        options = {train_dir : common(train_dir) for train_dir in train_dirs}
        enum(options, command, 'Select a train subdirectory', back = back)
    return _select_train_dir    
    
def detect_feature_importance(train_dir):
    """ Adds the "info" entry when selecting a train_dir for the feature-importance command """
    if 'importance' in listdir(os.path.join('train', train_dir)):
        return '(already exists)'
    return ''

def detect_evaluate(train_dir):
    """ Adds the "info" entry when selecting a train_dir for the evaluate command """
    path = os.path.join(working_directory, 'train', train_dir, 'evaluate')
    if train_dir in listdir('train') and os.path.exists(path):
        ret = []
        for file in listdir(path):
            ret.append(file.replace('.csv', ''))
        return f'({", ".join(ret)})'
    return ''

""" genolearn-setup """

def setup():
    """ Sets the data directory and proceeds to set the metadata file path """
    def _setup_meta(dir):
        csvs    = [csv for csv in os.listdir(dir) if csv.endswith('.csv') or csv.endswith('.txt')]
        if len(csvs) == 0:
            return print(f'no csv files found in {os.path.abspath(dir)}!')
        options = {csv : {'func' : lambda csv : _setup(dir, csv)} for csv in csvs}
        enum(options, 'setup', 'Select metadata csv for setup', back = setup)
    if working_directory == os.path.abspath('.'):
        return print('current directory is already setup')
    dirs = [dir for dir in os.listdir() if os.path.isdir(dir) and not dir.startswith('_') and not dir.startswith('.')]
    options = {'.' : {'func' : _setup_meta, 'info' : '(current directory)'}}
    for dir in dirs:
        options[dir] = {'func' : _setup_meta}
    try:
        enum(options, 'setup', f'Select data directory for setup within {os.path.abspath(".").replace(os.path.expanduser("~"), "~")}', back = exit)
    except KeyboardInterrupt:
        print()

def _setup(data_dir, meta):
    """ Sets the current working directory and metadata file within it """
    df = pd.read_csv(os.path.join(data_dir, meta))
    if len(df.columns) < 2:
        return print(f'"{meta}" does not contain enough columns')
    config = dict(data_dir = data_dir, meta = meta, history = [f'setup ({__version__})'])
    with open('.genolearn', 'w') as file:
        print(json.dumps(config, indent = 4), file = file, end = '')
    print('setup complete in current directory')


""" genolearn-clean """

def clean():
    """ Deletes all GenoLearn generated files upon user confirmation """
    if working_directory and os.path.exists(working_directory):
        option = dict(confirm = dict(func = _clean, info = 'this cannot be undone'))
        try:
            enum(option, 'clean', 'confirm deletion of all GenoLearn generated files in {working_directory.replace(os.path.expanduser("~"), "~")}?', back = exit)
        except:
            print()
    else:
        print('unknown working directory - either cd into working directory then re-execute genolearn-clean or check if directory already clean')

def _clean():
    """ Deletes all GenoLearn generated files """
    os.chdir(working_directory)
    for dir in ['evaluate', 'feature-selection', 'meta', 'model', 'preprocess', 'train']:
        if dir in ls:
            rmtree(dir)
    hidden = '.genolearn'
    if hidden in ls:
        os.remove(hidden)
    path = os.path.join(os.path.dirname(__file__), 'wd')
    with open(path) as f:
        wd = f.read()
    if wd == working_directory:
        os.remove(path)
    print(f'cleaned {working_directory}')

""" genolearn print """

def __print(name, limit = 5):
    """ Prints various files GenoLearn relies on """

    os.chdir(working_directory)

    if ' ' in name:
        name = name[:name.index(' ')]

    if active is None:
        return print('execute "genolearn config create" first')
    locations = ['meta', 'model']
    if name == 'None':
        for i, location in enumerate(locations):
            loc = location.replace(os.path.expanduser('~'), '~')
            if os.path.exists(location):
                print(f'{loc}\n  - ' + '\n  - '.join(os.listdir(location)))
            else:
                command = 'config model' if i else 'preprocess meta'
                print(f'{loc} (does not exist - execute "genolearn {command}" first)')
            if i < 1:
                print()
    if name == 'config':
        print('.genolearn config\n')
        with open('.genolearn') as f:
            log = json.load(f)
            log.pop('history')
            print(json.dumps(log, indent = 4))
    elif name == 'history':
        print('history\n')
        with open('.genolearn') as f:
            log = json.load(f)
            print('\n'.join(log['history']))
    elif 'meta' in name or 'model' in name:
        directory = os.path.dirname(name)
        dirname   = os.path.basename(directory)
        print(os.path.join(dirname, os.path.basename(name)), '\n')
        with open(name) as f:
            config = reduce(json.load(f), limit)
            print(json.dumps(config, indent = 4).replace('"', '').replace('...: ...', '...'))
    else:
        directory = os.path.dirname(name)
        dirname   = os.path.basename(directory)
        print(os.path.join(dirname, os.path.basename(name)), '\n')
        with open(name) as f:
            print(f.read())

def analyse(meta):
    """ Analyse preprocessed metadata for class label distribution and suggested target subset list """
    info  = dict(min_count  = dict(type = click.IntRange(0), default = 10),
                 proportion = dict(type = click.BOOL, default = False))

    print(f'{PRE}\n\nCommand: print metadata analyse\n\nParameters to analyse "{meta}"\n')
    params = dict(meta = meta)
    params.update(prompt(info))

    params['meta'] = os.path.join('meta', params['meta'])
    
    from   genolearn.core.data import analyse
    
    os.chdir(working_directory)
    if os.path.exists(params['meta']):
        analyse(**params)
    else:
        print('execute genolearn preprocess first')

def head(meta, num = 10):
    """ Prints the first NUM rows of meta data """
    print(os.path.join('meta', meta), '\n')
    meta   = os.path.join(working_directory, 'meta', meta)
    from genolearn.core.data import head
    head(meta, num)


def tail(meta, num = 10):
    """ Prints the last NUM rows of meta data """
    print(os.path.join('meta', meta), '\n')
    meta   = os.path.join(working_directory, 'meta', meta)
    from genolearn.core.data import tail
    tail(meta, num)


def sample(meta, num = 10):
    """ Prints random NUM rows of meta data """
    print(os.path.join('meta', meta), '\n')
    meta   = os.path.join(working_directory, 'meta', meta)
    from genolearn.core.data import sample
    sample(meta, num)

def select_meta(command, pre, func):
    """ Wrapper for first selecting metadata """
    def _select_meta():
        options = {meta : {'func' : func} for meta in metas}
        enum(options, command, pre, back = metadata)
    return _select_meta

def metadata():
    """ metadata subcommand for the print command """
    def args(func):
        name    = func.__name__
        command = f'print metadata {name}'
        pre     = 'Select metadata'
        return command, pre, func

    options = {'analyse' : {'info' : 'analyses the metadata'                , 'func' : select_meta(*args(analyse))},
               'head'    : {'info' : 'prints the head of the metadata'      , 'func' : select_meta(*args(head))},
               'tail'    : {'info' : 'prints the tail of the metadata'      , 'func' : select_meta(*args(tail))},
               'sample'  : {'info' : 'prints random entries of the metadata', 'func' : select_meta(*args(sample))}}
    enum(options, 'print metadata', 'Prints metadata information', back = _print)

def _print():
    """ Prints various files GenoLearn relies on """
    func     = lambda command : __print(command)
    options  = dict(history = dict(info = 'history of significant GenoLearn commands', func = func),
                    config  = dict(info = 'current config', func = func))

    if 'meta' in ls:
        options['meta'] = dict(info = 'metadata information', func = metadata)

    if 'preprocess' in ls:               
        key          = os.path.join(working_directory, 'preprocess', 'preprocess.log')
        options[key] = dict(prompt = 'preprocess.log', info = '(preprocess)', func = func)

        if 'combine.log' in listdir('preprocess'):
            key          = os.path.join(working_directory, 'preprocess', 'combine.log')
            options[key] = dict(prompt = 'combine.log', info = '(preprocess)', func = func)

    for subdir in ['meta', 'feature-selection', 'model']:
        for file in listdir(subdir):
            if subdir != 'feature-selection' or file.endswith('.log'):
                key          = os.path.join(working_directory, subdir, file)
                options[key] = dict(prompt = file, info = f'({subdir})', func = func)

    for subdir in listdir('train'):
        for sub in listdir(os.path.join('train', subdir)):
            if sub.endswith('.log'):
                key          = os.path.join(working_directory, 'train', subdir, sub)
                options[key] = dict(prompt = sub, info = f'({os.path.join("train", subdir)})', func = func)

    enum(options, 'print', 'Select option to print', back = menu)

""" genolearn preprocess """

def preprocess_sequence_data():
    """ Select sequential data to preprocess """
    gzs     = [gz for gz in os.listdir(os.path.join(working_directory, active['data_dir'])) if gz.endswith('.gz')]
    if len(gzs) == 0:
        return print('no sequence data (*.gz) files found!')
    options = {}
    for gz in gzs:
        info = ''
        if 'preprocess' in listdir():
            log = read_log(os.path.join('preprocess', 'preprocess.log'))
            if os.path.basename(log['data']) == gz:
                info = '(already preprocessed)'
            if 'combine.log' in listdir('preprocess'):
                log = read_log(os.path.join('preprocess', 'combine.log'))
                if isinstance(log['data'], str):
                    data = [os.path.basename(log['data'])]
                else:
                    data = [os.path.basename(data) for data in log['data']]
                if gz in data:
                    info = '(already combined)'
        options[gz] = {'func' : preprocess_sequence, 'info' : info}
    enum(options, 'preprocess sequence', 'Select option to preprocess', back = preprocess)

def preprocess_sequence(data):
    """ Preprocesses sequential data """
    print(f'{PRE}\n\nparameters for "{data}" to preprocess\n')
    info   = dict(batch_size = dict(type = click.INT, default = None),
                  n_processes = dict(type = click.INT, default = None),
                  sparse = dict(type = click.BOOL, default = True),
                  dense = dict(type = click.BOOL, default = True),
                  verbose = dict(type = click.IntRange(1), default = 250000),
                  max_features = dict(type = click.IntRange(-1), default = None))

    params = dict(data = data)
    params.update(prompt(info))

    assert params['dense'] or params['sparse'], 'set either / both dense and sparse to True'

    params['data']         = os.path.join(working_directory, active['data_dir'], data).replace(os.path.expanduser('~'), '~')

    if params['max_features'] != None:
        params['verbose'] = params['max_features'] // 10

    from multiprocessing import cpu_count

    if params['batch_size'] == None:
        params['batch_size'] = min(resource.getrlimit(resource.RLIMIT_NOFILE)[1], 2 ** 14) # safety
    if params['n_processes'] == None:
        params['n_processes'] = cpu_count()

    print_dict('executing "preprocess sequence" with parameters:', params)

    from   genolearn.core.preprocess import preprocess as core_preprocess

    os.chdir(working_directory)
    core_preprocess('preprocess', **params)
    append(f'preprocess sequence ({data})')

def preprocess_combine_data():
    """ Select sequential data to preprocess and combine with already preprocessed data """
    with open(os.path.join(working_directory, '.genolearn')) as f:
        log = json.load(f)
        preprocessed = []
        for line in log['history']:
            if line.startswith('preprocess sequence'):
                preprocessed.append(line[21:-1])
    if 'combine.log' in listdir('preprocess'):
        log  = read_log(os.path.join('preprocess', 'combine.log'))
        if isinstance(log['data'], str):
            preprocessed.append(os.path.basename(log['data']))
        else:
            preprocessed += [os.path.basename(file) for file in log['data']]
    options = {}
    for data in listdir(active['data_dir']):
        if data.endswith('.gz') and data not in preprocessed:
            options[data] = {'func' : preprocess_combine}
    if len(options) == 0:
        return print('no un-preprocessed sequence data (.gz) files found!')
    enum(options, 'preprocess combine' ,'Select option to preprocess and combine', back = preprocess)

def preprocess_combine(data):
    """ Preprocess sequential data and combine with already preprocessed data """
    print(f'preprocess {data}')
    info = dict(batch_size   = dict(type = click.INT, default = None),
                n_processes  = dict(type = click.INT, default = None),
                verbose      = dict(type = click.INT, default = 250000))

    params = dict(data = data)
    params.update(prompt(info))

    meta   = read_log(os.path.join('preprocess', 'preprocess.log'))
    params['max_features'] = meta['max_features']

    params['data'] = os.path.join(working_directory, active['data_dir'], data).replace(os.path.expanduser('~'), '~')

    if params['max_features'] != None:
        params['verbose'] = params['max_features'] // 10
        
    from multiprocessing import cpu_count

    if params['batch_size'] == None:
        params['batch_size'] = min(resource.getrlimit(resource.RLIMIT_NOFILE)[1], 2 ** 14) # safety
    if params['n_processes'] == None:
        params['n_processes'] = cpu_count()

    print_dict('executing "preprocess combine" with parameters:', params)

    from   genolearn.core.preprocess import combine

    if 'combine.log' in listdir('preprocess'):
        log    = read_log(os.path.join('preprocess', 'combine.log'))
        PARAMS = get_params().copy()
        if isinstance(log['data'], str):
            PARAMS['data']  = [log['data'], params['data'].replace(os.path.expanduser('~'), '~')]
        else:
            PARAMS['data'] += [log['data']]
        set_params(PARAMS)
    os.chdir(working_directory)
    combine('preprocess', **params)
        
    append(f'preprocess combine ({data})')

def preprocess_meta():
    """ Preprocesses metadata """
    meta_path      = os.path.join(working_directory, active['data_dir'], active['meta'])
    print(f'{PRE}\n\nCommand: preprocess meta\n\nParameters for preprocessing "{os.path.basename(active["meta"])}"\n')

    meta_df        = pd.read_csv(meta_path).applymap(str)
    valid_columns  = set(meta_df.columns)
    
    output         = click.prompt('output       ', type = click.STRING, default = 'default')
    identifier     = click.prompt('identifier             ', type = click.Choice(valid_columns), show_choices = False)

    valid_columns -= set([identifier])

    target         = click.prompt('target                 ', type = click.Choice(valid_columns), show_choices = False)

    valid_columns -= set([target])
    valid_columns |= set(['None'])

    group          = click.prompt('group           ', type = click.Choice(valid_columns), show_choices = False, default = 'None')

    if group != 'None':
        groups       = set(sorted(set(meta_df[group])))
        train_values = _prompt('train group values*    ', type = click.Choice(groups), default = None, default_option = False, multiple = True)
        groups      -= set(train_values)
        test_values  = _prompt('test  group values*    ', type = click.Choice(groups), default = None, default_option = False, multiple = True)
        ptrain       = None
    else:
        train_values = ['Train']
        test_values  = ['Test']
        ptrain       = click.prompt('proportion train', type = click.FloatRange(0., 1.), default = 0.75)

    from genolearn.core.preprocess import preprocess_meta

    os.chdir(working_directory)
    preprocess_meta(output, meta_path, identifier, target, group, train_values, test_values, ptrain)
    append(f'preprocess meta ({output})')

def preprocess():
    """ Preprocess command """
    options = {'sequence' : {'info' : 'preprocesses sequence data',
                             'func' : preprocess_sequence_data},
               'combine'  : {'info' : 'preprocesses sequence data and combines to previous preprocessing',
                             'func' : preprocess_combine_data},
               'meta'     : {'info' : 'preprocesses meta data',
                             'func' : preprocess_meta}}
    enum(options, 'preprocess', 'Select a preprocess subcommand', k = None if check_history('preprocess sequence') else 1)

""" genolearn feature-selection """

def _feature_selection():
    """ Wrapper for first selecting metadata and then feature-selection method """

    path = 'feature-selection'

    def detect(meta, method = None):
        ret = []
        if os.path.exists(os.path.join(working_directory, path)):
            for file in listdir(path):
                if file.endswith('.log'):
                    log = read_log(os.path.join(path, file))
                    if method:
                        if (log['meta'], log['method']) == (meta, method):
                            return '(already exists)'
                    elif log['meta'] == meta:
                        ret.append(log['method'])
        return f'({", ".join(ret)})' if ret else ''
        
    def _select_feature_selection(meta):
        fisher  = os.path.join(os.path.dirname(__file__), 'core', 'feature_selection', 'fisher.py')
        func    = lambda method : feature_selection(meta, method)
        exists  = detect(meta, 'fisher')
        options = {fisher : {'prompt' : 'fisher', 'info' : exists if exists else 'Fisher Score for Feature Selection', 'func' : func}}
        for dir in set([working_directory, os.path.abspath('.')]):
            for file in os.listdir(dir):
                if file.endswith('.py'):
                    py  = file.replace('.py', '')
                    key = os.path.join(dir, file)
                    options[key] = {'prompt' : py, 'func' : func, 'info' : detect(meta, py)}        
        enum(options, 'feature-selection', f'Select a feature selection method to use for "{meta}" metadata', back = _feature_selection)

    options = {meta : {'func' : _select_feature_selection, 'info' : detect(meta)} for meta in metas}
    enum(options, 'feature-selection', 'Select a metadata file')
            
def feature_selection(meta, module):
    """ Computes Feature Selection (Fisher by default) """
    print(f'{PRE}\n\nCommand: feature-selection\n\nParameters for feature selection using "{meta}" meta with "{os.path.basename(module)[:-3]}" method\n')
    py     = os.path.basename(module).replace('.py', '')
    params = dict(meta = meta, method = py, module = module.replace(os.path.expanduser('~'), '~'))
    info   = dict(name = dict(default = f'{params["meta"]}-{py}', type = click.STRING))
    
    params.update(prompt(info))

    print_dict('executing "feature-selection" with parameters:', params)

    params['log'] = f"{params['name']}.log"

    from   genolearn.core.feature_selection import feature_selection
    os.chdir(working_directory)
    feature_selection(**params)
    append(f'feature-selection ({params["name"]})')

""" genolearn model-config """

classifiers = dict(
                  logistic_regression = \
                  dict(model = 'LogisticRegression',
                          config_name = dict(type = click.STRING, default = 'logistic-regression'),
                          penalty = dict(type = click.Choice(['l1', 'l2', 'elasticnet', 'none']), default = 'l2', show_choices = True),
                          dual = dict(type = click.BOOL, default = False),
                          tol = dict(type = click.FloatRange(1e-8), default = 1e-4),
                          C = dict(type = click.FloatRange(1e-8), default = 1.),
                          fit_intercept = dict(type = click.BOOL, default = True),
                          class_weight = dict(type = click.Choice(['balanced', 'None']), default = 'None', show_choices = True),
                          random_state = dict(type = click.INT, default = None),
                          solver = dict(type = click.Choice(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']), default = 'lbfgs', show_choices = True),
                          max_iter = dict(type = click.IntRange(1), default = 100),
                          multi_class = dict(type = click.Choice(['auto', 'ovr', 'multinomial']), default = 'auto', show_choices = True),
                          n_jobs = dict(type = click.IntRange(-1), default = -1),
                          l1_ratio = dict(type = click.FloatRange(0), default = 1.)),
                  random_forest = \
                  dict(model = 'RandomForestClassifier',
                          config_name = dict(type = click.STRING, default = 'random-forest'),
                          n_estimators = dict(type = click.IntRange(1), default = 100),
                          criterion = dict(type = click.Choice(['gini', 'entropy', 'log_loss']), default = 'gini', show_choices = True),
                          max_depth = dict(type = click.IntRange(1), default = None),
                          min_samples_split = dict(type = click.IntRange(1), default = 2),
                          min_samples_leaf = dict(type = click.IntRange(1), default = 1),
                          min_weight_fraction_leaf = dict(type = click.FloatRange(0., 0.5), default = 0.),
                          max_features = dict(type = click.Choice(['sqrt', 'log2', 'None']), default = 'sqrt', show_choices = True),
                          max_leaf_nodes = dict(type = click.IntRange(1), default = None),
                          min_impurity_decrease = dict(type = click.FloatRange(0), default = 0.),
                          bootstrap = dict(type = click.BOOL, default = True),
                          oob_score = dict(type = click.BOOL, default = False),
                          n_jobs = dict(type = click.IntRange(-1), default = -1),
                          random_state = dict(type = click.INT, default = None),
                          class_weight = dict(type = click.Choice(['balanced', 'balanced_subsample', 'None']), default = None, show_choices = True))
                  )

def _model(name):
    """ Given a model name, prompts user for hyperparameter settings """
    classifier  = classifiers[name]
    params      = {'model' : classifier.pop('model')}
    print(f'{PRE}\n\nCommand: model-config\n\nParameters for {params["model"]}\n')
    params.update(prompt(classifier))
    config_name = params.pop('config_name')
    path        = os.path.join(working_directory, 'model')
    os.makedirs(path, exist_ok = True)
    with open(os.path.join(path, config_name), 'w') as file:
        print(json.dumps(params, indent = 4), file = file)
    print(f'generated "{config_name}" in {path}')
    append(f'model ({config_name})')

def model_config():
    """ Prompts user for model to use before prompting for hyperparameter settings """
    models  = ['logistic_regression', 'random_forest']
    options = {model : {'func' : _model} for model in models}
    enum(options, 'model-config', 'Select a model to configure')

""" genolearn train """

def detect_train(meta, feature_selection = None, model_config = None):
    """ Adds the "info" entry when selecting a metadata file for the train command """
    ret    = []
    num    = 1 + bool(feature_selection) + bool(model_config)
    target = (meta, feature_selection, model_config)[:num]
    if 'train' in ls:
        for train_dir in listdir('train'):
            log = read_log(os.path.join('train', train_dir, 'train.log'))
            if (log['meta'], log['feature_selection'], log['model_config'])[:num] == target:
                ret.append(train_dir)
    return f'({", ".join(ret)})' if ret else ''

def _train():
    def _select_model_config(meta_feature_selection):
        meta, feature_selection = meta_feature_selection
        options = {}
        for model_config in listdir('model'):
            func = lambda model_config : train(meta, feature_selection, model_config)
            info = detect_train(meta, feature_selection, model_config)
            options[model_config] = {'func' : func, 'info' : info}
        pre_text = f'Select a model config to use with "{meta}" metadata and "{feature_selection}" feature selection to train'
        enum(options, 'train', pre_text, back = _train)
    options = {}
    for selection in listdir('feature-selection'):
        if selection.endswith('.log'):
            log  = read_log(os.path.join('feature-selection', selection))
            meta = log['meta']
            selection = selection.replace('.log', '')
            options[(meta, selection)] = {'prompt' : selection, 'info' : f'("{meta}" metadata)', 'func' : _select_model_config}
    enum(options, 'train', 'Select a feature selection file to use to train')

def train(meta, feature_selection, model_config):
    """ Given a preprocessed metadata file, trains model(s) and save outputs to the train subdirectory within the working directory """
    params = locals().copy()
    print(f'{PRE}\n\nCommand: train\n\nTrain parameters for metadata "{meta}" with feature-selection "{feature_selection}" and model config "{model_config}"\n')

    default = f'{feature_selection}-{model_config}'
    group   = ['Train', 'Test']
    with open(os.path.join(working_directory, 'meta', meta)) as f:
        meta_ = json.load(f)
        if set(meta_['group']) != {'Train', 'Test'}:
            group = list(meta_['group']) + group
        
    from genolearn.metrics import _metrics

    choice  = click.Choice(sorted(set(_metrics) - {'count'}))
    info    = dict(output_dir = dict(type = click.Path(), default = default),
                   num_features = dict(default = 1000, type = click.IntRange(1), multiple = True),
                   min_count = dict(default = 0, type = click.IntRange(0)),
                   target_subset = dict(default = 'None', type = click.Choice(group)),
                   metric = dict(default = 'f1_score', type = choice, show_choices = False),
                   aggregate_func = dict(default = 'weighted_mean', type = click.Choice(['mean', 'weighted_mean'])))

    params.update(prompt(info))

    os.chdir(working_directory)
    os.makedirs('train', exist_ok = True)
    
    params['output_dir'] = os.path.join('train', params['output_dir'])
    
    if os.path.exists(params['output_dir']):
        rmtree(params['output_dir'])

    if isinstance(params['num_features'], int):
        params['num_features'] = [params['num_features']]

    print_dict('executing "genolearn train" with parameters:', params)

    from genolearn.core.train import train
    train(**params)
    append(f'train ({os.path.basename(params["output_dir"])})')

""" genolearn feature-importance """

def feature_importance(train_dir):
    """ Given a training directory, computes the Feature Importance and outputs an Importance subdirectory """
    params = dict(train_dir = train_dir)

    print_dict('executing "genolearn feature-importance" with parameters:', params)
    
    params['train_dir'] = os.path.join('train', params['train_dir'])
    params['model']     = os.path.join(params['train_dir'], 'model.pickle')
    params['output']    = os.path.join(params['train_dir'], 'importance')
    
    os.chdir(working_directory)

    log = read_log(os.path.join(params.pop('train_dir'), 'train.log'))
    params['feature_selection'] = log['feature_selection']
    params['meta'] = log['meta']

    from   genolearn.core.feature_importance import feature_importance

    feature_importance(**params)
    append(f'feature-importance ({train_dir})')

""" genolearn evaluate """

def evaluate(train_dir):
    """  Given a training directory, evaluates a model on user prompted inputs and outputs to the evaluate subdirectory within the working directory """
    print(f'{PRE}\n\nCommand: evaluate\n\nEvaluate parameters for "{train_dir}"\n')

    path = os.path.join(working_directory, 'train', train_dir)
    log  = read_log(os.path.join(path, 'train.log'))
    meta = log['meta']

    with open(os.path.join(working_directory, 'meta', meta)) as f:
        meta   = json.load(f)
        if set(meta['group']) == {'Train', 'Test'}:
            groups = []
        else:
            groups = list(meta['group'])
        groups += ['Train' ,'Test', 'unlabelled']

    info   = dict(output    = dict(prompt = 'output filename', type = click.Path()),
                  values    = dict(prompt = 'group values', type = click.Choice(groups), multiple = True))

    params = dict(train_dir = train_dir)
    params.update(prompt(info))

    log = read_log(os.path.join(path, 'train.log'))
    for key in ['meta', 'feature_selection']:
        params[key] = log[key]

    log = read_log(os.path.join(path, 'params.json'))
    params['num_features'] = log['num_features']

    print_dict('executing "evaluate" with parameters:', params)

    log = read_log(os.path.join(path, 'encoding.json'))
    params['encoder'] = log

    path   = os.path.join(path, 'evaluate')

    os.makedirs(path, exist_ok = True)

    params['output'] = os.path.join(path, params['output'])

    if not params['output'].endswith('.csv'):
        params['output'] = params['output'] + '.csv'

    from   genolearn.core.evaluate import evaluate

    data_config = dict(working_dir = working_directory, meta_file = params.pop('meta'))

    params['data_config'] = data_config

    params['model'] = os.path.join(os.path.join(working_directory, 'train', params.pop('train_dir')), 'model.pickle')

    os.chdir(os.path.dirname(path))

    evaluate(**params)
    append(f'evaluate ({train_dir} {os.path.basename(params["output"]).replace(".csv", "")})')

def menu():
    """ Main menu for GenoLearn """

    _feature_importance = select_train_dir('feature-importance', feature_importance, menu, detect_feature_importance)
    _evaluate           = select_train_dir('evaluate', evaluate, menu, detect_evaluate)

    options             = {'print'              : {'info' : 'prints various GenoLearn generated files',
                                                   'func' : _print},
                           'preprocess'         : {'info' : 'preprocess data into an easier format for file reading',
                                                   'func' : preprocess},
                           'feature-selection'  : {'info' : 'computes a feature selection method for later training',
                                                   'func' : _feature_selection},
                           'model-config'       : {'info' : 'creates a machine learning model config',
                                                   'func' : model_config},
                           'train'              : {'info' : 'trains a machine learning model',
                                                   'func' : _train},
                           'feature-importance' : {'info' : 'computes the model feature importances',
                                                   'func' : _feature_importance},
                           'evaluate'           : {'info' : 'evaluates a trained model on an input dataset',
                                                   'func' : _evaluate}}

    # set k to not truncate options initially
    k = None

    # set k = 1 if not in working directory or has not executed setup
    if not check_working_directory():
        return print('unknown working directory - cd into the working directory and execute genolearn again or execute ' \
                     'genolearn-setup to setup this directory as the working directory')
        

    # if a command has not been previously executed truncate options to not include other commands that rely on it.
    else:
        for i, command in enumerate(['preprocess meta', 'feature-selection', 'model', 'train'], 2):
            if not check_history(command):
                k = i
                break
    try:
        enum(options, '', k = k, back = exit)
    except KeyboardInterrupt:
        print()
    except click.exceptions.Abort:
        print()