from   biolearn.logger import Computing, msg
import numpy as np

def fisher(X, Y, name = None, indent = 0):
    # https://stats.stackexchange.com/questions/277123/fisher-score-feature-selection-implementation
    if name is not None: msg(f'computing fisher scores for "{name}"', indent = indent)
    mu  = X.mean(axis = 0)
    Yj  = [np.where(Y[:,j]) for j in range(Y.shape[1])]
    N   = np.array(list(map(len, Yj)))
    M   = []
    V   = []
    for i, yj in enumerate(Yj, 1):
        with Computing(f'statistics for class {i} of {len(Yj)}'):
            m  = X[yj].mean(axis = 0)
            m2 = np.square(X[yj]).mean(axis = 0)
            M.append(m)
            V.append(m2 - np.square(m))
        
    num      = np.einsum('n,nm->m', N, np.square(M - mu))
    den      = np.einsum('n,nm->m', N, V)
    
    # Fisher score is 0 when den is 0 (avoids divide by 0 error)
    idx      = np.where(den == 0)
    num[idx] = 0
    den[idx] = 1
    
    ret      = num / den
    if name is not None: msg(f'computed fisher score for "{name}"', delete = 1, indent = indent)
    return ret

def _fisher(x, y):
    u = set(y)
    c = len(u)
    m = x.mean()
    top = 0
    bot = 0
    for k in u:
        mask = y == k
        n_j  = mask.sum()
        top += n_j * np.square(x[mask].mean() - m)
        bot += n_j * np.square(x[mask].std())
    return top / bot if bot else 0
        
def weighted_fisher(X, Y, name = None, indent = 0):
    # https://stats.stackexchange.com/questions/277123/fisher-score-feature-selection-implementation
    if name is not None: msg(f'computing weighted fisher scores for "{name}"', indent = indent)
    W   = 1 / (Y.mean(axis = 0)[None,:] * np.ones((len(Y), 1)) / len(Y)).sum(axis = 1)
    mu  = np.einsum('nm,n->m', X, W) # X.mean(axis = 0)
    Yj  = [np.where(Y[:,j]) for j in range(Y.shape[1])]
    N   = np.array(list(map(len, Yj)))
    M   = []
    V   = []
    for i, yj in enumerate(Yj, 1):
        with Computing(f'statistics for class {i} of {len(Yj)}'):
            m  = X[yj].mean(axis = 0)
            m2 = np.square(X[yj]).mean(axis = 0)
            M.append(m)
            V.append(m2 - np.square(m))
        
    num      = np.sum(np.square(M - mu), axis = 0) # np.einsum('n,nm->m', N, np.square(M - mu))
    den      = np.sum(V, axis = 0) # np.einsum('n,nm->m', N, V)
    
    # Fisher score is 0 when den is 0 (avoids divide by 0 error)
    idx      = np.where(den == 0)
    num[idx] = 0
    den[idx] = 1
    
    ret      = num / den
    if name is not None: msg(f'computed weighted fisher score for "{name}"', delete = 1, indent = indent)
    return ret