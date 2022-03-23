if __name__ == '__main__':

    from   genolearn.logger  import print_dict, msg
    from   genolearn         import DataLoader, utils, _data

    from   argparse          import ArgumentParser, RawTextHelpFormatter

    import numpy  as np

    import shutil
    import json
    import gzip
    import re
    import os

    description = \
    r"""
    Generates an ordered list of features and meta information.
    """

    parser = ArgumentParser(description = description, formatter_class = RawTextHelpFormatter)

    parser.add_argument('output')
    parser.add_argument('path')
    parser.add_argument('meta_path')
    parser.add_argument('identifier')
    parser.add_argument('target')
    parser.add_argument('values', nargs = '+')
    parser.add_argument('-group', default = None)
    parser.add_argument('-method', default = 'fisher')
    parser.add_argument('--sparse', default = False, action = 'store_true')

    args   = parser.parse_args()
    params = dict(args._get_kwargs())
    print_dict('executing "genolearn.feature_selection" with parameters:', params)

    dataloader = DataLoader(args.path, args.meta_path, args.identifier, args.target, args.group, args.sparse)

    if args.method == 'fisher':
        from genolearn.feature_selection.fisher import fisher_score

        scores = fisher_score(dataloader, args.values)
    
    else:
        if f'{args.method}.py' in os.listdir():
            from   genolearn.feature_selection.base import _feature_selection
            import importlib

            module       = importlib.import_module(args.method)

            variables    = dir(module)

            for name in ['init', 'inner_loop', 'outer_loop']:
                assert name in variables
                
            force_sparse = module.force_sparse if 'force_sparse' in variables else False
            force_dense  = module.force_sparse if 'force_dense'  in variables else False

            scores       = _feature_selection(dataloader, module.init, module.inner_loop, module.outer_loop, args.values, force_dense, force_sparse)

    np.savez_compressed(f'{args.path}/feature-selection/{args.output}', **scores)

    msg('executed "genolearn.feature_selection"')
