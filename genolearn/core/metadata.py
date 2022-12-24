import pandas as pd
import json

def _load_df(meta):
    
    with open(meta) as f:
        meta = json.load(f)

    m, g, t, s = meta['identifiers'], [], [], []
    for identifier in m:
        t.append(meta['search'][identifier])
        for group, identifiers in meta['group'].items():
            if identifier in identifiers:
                g.append(group)
                continue

        s.append('train' if any(identifier in meta['group'][group] for group in meta['Train']) else 'val')

    df = pd.DataFrame(list(*zip(m ,g, t, s)), columns = ['identifier', 'group', 'target', 'train/val'])
    df.index += 1
    return df
    
def _count(meta):
    df    = _load_df(meta)
    count = pd.DataFrame(dtype = int)
    for target in sorted(set(df['target'])):
        for group in sorted(set(df['group'])):
            count.loc[target, group] = ((df['target'] == target) & (df['group'] == group)).sum()

    if 'train' in count.columns:
        count.columns  = [column.title() for column in count.columns]
    else:
        count['Train'] = count[list(meta['Train'])].sum(axis = 1)
        count['Val'  ] = count[list(meta['Val'  ])].sum(axis = 1)

    count['Total']     = count['Train'] + count['Val']
    count.loc['Total'] = count.sum(axis = 0)
    return count

def display(drop = False):
    def outer(func):
        def inner(meta):
            if drop:
                df = func(_load_df(meta), 10) # arbitrary setting of 10 rows printed
                if (df['group'] == df['train/val']).all():
                    df.drop('group', axis = 1, inplace = True)
            else:
                df = func(_count(meta)).applymap(int)
            print(f'genolearn command : print metadata {func.__name__}')
            print(f'metadata          : {meta.split("/")[-1]}', '\n')
            print(df)
        return inner
    return outer
    
@display()
def count(df):
    return df

@display()
def proportion(df):
    df.iloc[:-1] /= df.iloc[-1]
    df.iloc[-1]  /= df.iloc[-1,-1]
    df           *= 100
    return df.round(0)

@display(True)
def head(df, num):
    return df.head(num)
    
@display(True)
def tail(df, num):
    return df.tail(num)

@display(True)
def sample(df, num):
    return df.sample(num)
