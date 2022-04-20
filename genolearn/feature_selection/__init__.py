from   genolearn.logger import msg

import numpy as np

def base_feature_selection(dataloader, init, inner_loop, outer_loop, values, force_dense = False, force_sparse = False):
    """
    base feature selection function

    Parameters
    ----------
        dataloader : str
            genolearn.dataloader.DataLoader object.

        init : str
            Initialise function to generate and the associated ``args`` and ``kwargs`` for ``inner_loop`` and ``outer_loop`` functions.

        inner_loop : str
            Inner loop function to be executed on a given (x, y) pair.
        
        outer_loop : str
            Outer loop function to be executed for each value in ``values``.
        
        values : list, *default=[]*
            Set of values to use when looping over the generator method of the ``DataLoader`` object.

        force_dense : bool, *default=False*
            Identify if computations should be forced to dense computations.

        force_dense : bool, *default=False*
            Identify if computations should be forced to sparse computations.
        
    Notes
    -----

    If it is intended to execute the ``inner_loop`` function on all examplars and the ``outer_loop`` function once at the end, ``values`` should be set such that it evaluates to ``False``.
    """
    ret          = {}
    args, kwargs = init(dataloader)
    if values:
        for value in values:
            for i, (x, label) in enumerate(dataloader.generator(value, force_dense = force_dense, force_sparse = force_sparse), 1):
                inner_loop(ret, i, x, label, value, *args, **kwargs)
            outer_loop(ret, i, value, *args, **kwargs)
    else:
        values = dataloader.meta.index
        for i, (x, label) in enumerate(dataloader.generator(*values, force_dense = force_dense, force_sparse = force_sparse), 1):
            inner_loop(ret, i, x, label, 'all', *args, **kwargs)
        outer_loop(ret, i, 'all', *args, **kwargs)
    return ret