from   genolearn.logger import Computing, msg
import os

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


def _fisher_score(dataloader, meta, target_column, index_column, group_column):
    if group_column is not None:
        ix_group_target = meta.loc[:,[index_column, group_column, target_column]].copy()
        scores = {group : {target : [0, 0, 0] for target in set(ix_group_target.loc[:,target_column])} for group in set(ix_group_target.loc[:,group_column])}
        for i in ix_group_target.index:
            ix, group, target = ix_group_target.loc[i]
            x = dataloader.load_X(ix)
            scores[group][target][0] += x
            scores[group][target][1] += x.power(2)
            scores[group][target][2] += 1
        return scores
    else:
        ix_target = meta.loc[:,[index_column, target_column]].copy()
        scores = {target : [0, 0, 0] for target in set(ix_target.loc[:,target_column])}
        for i in ix_target.index:
            ix, target = ix_target.loc[i]
            x = dataloader.load_X(ix)
            scores[target][0] += x
            scores[target][1] += x.power(2)
            scores[target][2] += 1
        return scores

def fisher_score_by_year(dataloader, meta, target_column, index_column, group_column):
    scores = _fisher_score(dataloader, meta, target_column, index_column, group_column)
    S      = {}

    for start in range(min(scores), max(scores)):
        years = range(start, max(scores))
        sub   = [0, 0]
        N     = {}
        for year in years:
            for label in scores[year]:
                if scores[year][label][2] == 0: continue
                sub[0] += scores[year][label][0].A
                sub[1] += scores[year][label][2]
                if label not in N:
                    N[label] = 0
                N[label] += scores[year][label][2]
        
        mean = sub[0] / sub[1]

        top  = 0
        bot  = 0

        for year in years:
            for label in scores[year]:
                if scores[year][label][2] == 0: continue
                mean_ij = scores[year][label][0].A / scores[year][label][2]
                mom2_ij = scores[year][label][1].A / scores[year][label][2]
                var_ij  = mom2_ij - mean_ij
                top += N[label] * np.square(mean_ij - mean)
                bot += N[label] * var_ij
        
        S[str(start)] = np.divide(top, bot, where = bot > 0)

        np.savez_compressed(os.path.join(dataloader.path, 'feature-selection', 'fisher-by-year.npz'), **S)
