def base_feature_selection(method, dataloader, init, loop, post):
    """
    base feature selection function

    Parameters
    ----------
        method : str
            Method name.

        dataloader : str
            genolearn.dataloader.DataLoader object.

        init : str
            Initialise function to generate and the associated ``args`` and ``kwargs`` for ``inner_loop`` and ``outer_loop`` functions.

        inner_loop : str
            Inner loop function to be executed on a given (x, y) pair.
        
        outer_loop : str
            Outer loop function to be executed for each value in ``values``.
    """
    from   genolearn.logger import msg, Waiting
    args, kwargs = init(dataloader)
    n            = sum(map(len, (dataloader.meta['group'][group] for group in dataloader.meta['Train'])))
    for i, (x, label) in enumerate(dataloader.generator('Train'), 1):
        msg(f'{method} : {i:,d} of {n:,d}', inline = True)
        loop(i, x, label, 'Train', *args, **kwargs)
    
    with Waiting(f'{method} : {i:,d} (computing scores)', f'{method} : {i:,d} (computed scores)'):
        ret = post(i, 'Train', *args, **kwargs)

    return ret

def feature_selection(name, meta, method, module, log):

    from   genolearn.logger     import msg, Writing
    from   genolearn.dataloader import DataLoader
    from   genolearn            import utils

    import numpy  as np
    import os

    dataloader = DataLoader(meta, utils.working_directory)
    
    os.makedirs('feature-selection', exist_ok = True)

    variables  = {}
    with open(os.path.expanduser(module)) as f:
        exec(f.read(), {}, variables)

    save_path    = os.path.join('feature-selection', name)
    funcs        = ['init', 'loop', 'post']

    for name in funcs:
        assert name in variables
        
    params       = {func : variables.get(func) for func in funcs}
    scores       = base_feature_selection(method, dataloader, **params)

    with Writing(save_path, inline = True):
        np.savez_compressed(save_path, scores)
        os.rename(f'{save_path}.npz', save_path)

    utils.create_log(log, 'feature-selection')

    msg(f'executed "genolearn feature-selection"')
