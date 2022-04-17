if __name__ == '__main__':
    from   genolearn.feature_selection import base_feature_selection
    from   genolearn.logger  import print_dict, msg, Writing
    from   genolearn.dataloader import DataLoader
    from   genolearn         import utils

    from   argparse          import ArgumentParser, RawTextHelpFormatter
    import importlib
    import pkgutil

    import numpy  as np
    import os

    description = \
    r"""
    Generates an ordered list of features and meta information.

    Example
    =======

    >>> # fisher score default
    >>> python -m genolearn.feature_selection fisher-scores.npz data raw-data/meta-data.csv Accession Regions 2014 2015 2016 2017 2018 2019 -group Year

    >>> # custom (expected custom.py)
    >>> python -m genolearn.feature_selection custom-scores.npz data raw-data/meta-data.csv Accession Regions 2014 2015 2016 2017 2018 2019 -group Year -method custom

    """

    parser = ArgumentParser(description = description, formatter_class = RawTextHelpFormatter)

    parser.add_argument('output',     help = 'output file name')
    parser.add_argument('path'  ,     help = 'path to preprocessed directory')
    parser.add_argument('meta_path',  help = 'path to meta file')
    parser.add_argument('identifier', help = 'column of meta data denoting the identifier')
    parser.add_argument('target',     help = 'column of meta data denoting the target')
    parser.add_argument('values', nargs = '*', help = 'incremental identifiers (or groups) to perform feature selection on')
    parser.add_argument('-group', default = None, help = 'column of meta data denoting the grouping of labels')
    parser.add_argument('-method', default = 'fisher', help = 'either "fisher" for built in Fisher Score or a module name (see example)')
    parser.add_argument('-log', default = None, help = 'log file name')
    parser.add_argument('--sparse', default = False, action = 'store_true', help = 'if sparse loading of data is preferred')

    args   = parser.parse_args()
    params = dict(args._get_kwargs())

    print_dict('executing "genolearn.feature_selection" with parameters:', params)

    dataloader = DataLoader(args.path, args.meta_path, args.identifier, args.target, args.group, args.sparse)
    

    if f'{args.method}' in [module for _, module, _ in pkgutil.iter_modules(['genolearn/feature_selection']) if not module.startswith('__')]:

        module       = importlib.import_module(f'genolearn.feature_selection.{args.method}')

    elif f'{args.method}.py' in os.listdir():

        module       = importlib.import_module(args.method)

    else:
        raise Exception(f'"{args.method}.py" not in current directory!')

    variables    = dir(module)

    for name in ['init', 'inner_loop', 'outer_loop']:
        assert name in variables
        
    force_sparse = module.force_sparse if 'force_sparse' in variables else False
    force_dense  = module.force_dense  if 'force_dense'  in variables else False

    scores       = base_feature_selection(dataloader, module.init, module.inner_loop, module.outer_loop, args.values, force_dense, force_sparse)

    save_path    = f'{args.path}/feature-selection/{args.output}'
    with Writing(save_path, inline = True):
        np.savez_compressed(save_path, **scores)

    utils.create_log(f'{args.path}/feature-selection', f'log-{args.method}.txt' if args.log is None else args.log)

    msg('executed "genolearn.feature_selection"')
