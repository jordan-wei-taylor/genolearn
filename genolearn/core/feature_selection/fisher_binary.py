def init(dataloader):
    """
    Initialises statistics for the Fisher Score computation.
    """
    import numpy as np

    # encoding from class label to integer
    encode = {c : i for i, c in enumerate(set(dataloader.meta['targets']))}

    # class label count
    n  = np.zeros(dataloader.c)

    # global sum
    sg = np.zeros(dataloader.m)

    # class label sum
    s1 = np.zeros((dataloader.m, dataloader.c))

    # class label sum of squares
    s2 = np.zeros((dataloader.m, dataloader.c))

    args   = (encode, n, sg, s1, s2)  # encoder, counts, sum, by class sum, by class sum of squares
    kwargs = {}                       # no kwargs

    return args, kwargs

def loop(i, x, label, value, *args, **kwargs):
    """
    Incrementally updates count, global, and by class label statistics
    """

    encode, n, sg, s1, s2 = args
        
    y        = encode[label]

    # convert to binary 0 -> 0, {1, 2, 3, ...} -> 1
    x        = x > 0

    # increase count of label
    n[y]    += 1

    # increase global sum
    sg      += x

    # increase class label sum
    s1[:,y] += x

    # increase class label sum of squares
    s2[:,y] += x ** 2

def post(i, value, *args, **kwargs):
    """
    Computes the Fisher Score using statistics stored in ``*args``.
    """
    import numpy as np
    
    encode, n, sg, s1, s2 = args

    # convert global sum to global mean
    mu  = sg / n.sum()

    # convert to first and second moments
    m1  = np.divide(s1, n, where = n > 0)
    m2  = np.divide(s2, n, where = n > 0)

    # compute D and S as per www.genolearn.readthedocs.io/usage/feature-selection.html
    D   = np.square(m1 - mu.reshape(-1, 1)) # broadcast second dimension ((m, c) - (m, 1))
    S   = (m2 - np.square(m1))

    # numerator and denominator expressions for Fisher Score
    num = D @ n
    den = S @ n

    S   = np.divide(num, den, where = den > 0)

    return -S # return negative scores such that argsort returns largest to smallest
