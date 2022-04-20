from   genolearn.logger import msg

import numpy as np

def init(dataloader):
    """
    Initialises statistics for the Fisher Score computation for inner and outer loop functions.
    """
    # encoding from class label to integer
    encode = {c : i for i, c in enumerate(set(dataloader.meta[dataloader.target]))}

    # class label count
    n  = np.zeros(dataloader.c)

    # global sum
    sg = np.zeros(dataloader.m)

    # class label sum
    s1 = np.zeros((dataloader.c, dataloader.m))

    # class label sum of squares
    s2 = np.zeros((dataloader.c, dataloader.m))

    args   = (encode, n, sg, s1, s2)  # encoder, counts, sum, by class sum, by class sum of squares
    kwargs = {}                       # no kwargs

    return args, kwargs

def inner_loop(ret, i, x, label, value, *args, **kwargs):
    """
    Incrementally updates count, global, and by class label statistics
    """
    msg(f'{value} : {i}', inline = True)

    encode, n, sg, s1, s2 = args
    
    y      = encode[label]

    # increase count of label
    n[y]  += 1

    # increase global sum
    sg    += x

    # increase class label sum
    s1[y] += x

    # increase class label sum of squares
    s2[y] += np.square(x)

def outer_loop(ret, i, value, *args, **kwargs):
    """
    Computes the Fisher Score using statistics stored in ``*args``.
    """
    msg(f'{value} : {i} (computing fisher)', inline = True)

    encode, n, sg, s1, s2 = args

    # reshape for broadcasting
    n   = n.reshape(-1, 1)

    # convert global sum to global mean
    mu  = sg / n.sum()

    # convert to first and second moments
    m1  = np.divide(s1, n, where = n > 0)
    m2  = np.divide(s2, n, where = n > 0)

    # compute D and S as per www.genolearn.readthedocs.io/usage/feature-selection/index.html
    D   = np.square(m1 - mu)
    S   = m2 - np.square(m1)

    # reshape from 2D back to 1D
    n   = n.reshape(-1)

    # numerator and denominator expressions for Fisher Score
    num = n @ D
    den = n @ S

    S   = np.divide(num, den, where = den > 0)

    msg(f'{value} : {i} (completed)')

    ret[value] = S

# computations above require each "x" to be a numpy array and not a scipy sparse array
force_dense = True