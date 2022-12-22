def _analyse(meta, min_count, proportion):
    
    import pandas as pd

    check   = set(meta['group']) != {'Train', 'Test'}
    columns = list(meta['group']) + ['Train' ,'Test'] if check else ['Train' ,'Test']
    count   = pd.DataFrame(index = meta['targets'], columns = columns, data = 0)

    if check:
        for group in meta['group']:
            for identifier in meta['group'][group]:
                target = meta['search'][identifier]
                count.loc[target, group] += 1

    for key in ['Train', 'Test']:
        for group in meta[key]:
            for identifier in meta['group'][group]:
                count.loc[meta['search'][identifier], key] += 1

    count['Global']    = count['Train'] + count['Test']
    count.loc['Total'] = count.sum(axis = 0).copy()

    suggested_subset = count.index[:-1][count['Train'].values[:-1] >= min_count]

    if len(suggested_subset) == len(count.index):
        suggested_subset = None

    if proportion:
        count.iloc[:-1] /= count.iloc[:-1].sum(axis = 0)
        count.iloc[-1]   = count.iloc[-1] / count.iloc[-1,-1]
        count *= 100
        return count.round(2).applymap(lambda value : f'{value:6.2f}'), suggested_subset
    else:
        return count.applymap(int).applymap(lambda value : f'{value:6,d}'), suggested_subset
    
def analyse(meta, min_count, proportion):
    import json

    with open(meta) as f:
        meta = json.load(f)

    count, subset = _analyse(meta, min_count, proportion)

    maxlen1 = max(map(len, count.index))
    maxlen2 = max([max(map(len, count.values.flatten())), max(map(len, count.columns))])

    print(' ' * maxlen1 + ' | ' + ' | '.join(f'{col:>{maxlen2}s}' for col in count.columns))
    print('-' * (maxlen1 + 1) + '+' + '+'.join('-' * (maxlen2 + 2) for _ in range(count.shape[1])))
    for target in count.index:
        print(f'{target:{maxlen1}s} | ' + ' | '.join(list(count.loc[target])))

    if subset is not None:
        print('\nsuggested target subset:', ', '.join(subset))

def _load_df(meta):
    import pandas as pd
    import json

    with open(meta) as f:
        meta = json.load(f)

    g  = []
    t  = []
    for identifier in meta['identifiers']:
        t.append(meta['search'][identifier])
        for group, identifiers in meta['group'].items():
            if identifier in identifiers:
                g.append(group)
                continue

    df = pd.DataFrame()
    df['identifier'] = meta['identifiers']
    df[' group']     = g
    df[' target']    = t
    df.index        += 1
    return df

def head(meta, num):
    df = _load_df(meta)
    print(df.head(num))
    

def tail(meta, num):
    df = _load_df(meta)
    print(df.tail(num))

def sample(meta, num):
    df = _load_df(meta)
    print(df.sample(num))
