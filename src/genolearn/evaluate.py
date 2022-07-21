def main(args):
    from   genolearn.dataloader import DataLoader

    import joblib
    import pandas as pd

    model       = joblib.load(args.model)
    data_config = check_config(args.data_config)
    dataloader  = DataLoader(**data_config)

    if args.feature_selection and args.key:
        features = dataloader.load_feature_selection(args.feature_selection).rank(ascending = args.ascending)[args.key]
    else:
        features = None

    X   = dataloader.load_X(*args.values, features = features)

    df  = pd.DataFrame(index = dataloader.identifiers)
    df['hat'] = dataloader.decode(model.predict(X))
    
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)
        for i, name in enumerate(dataloader.encoder):
            df[f'P({name})'] = prob[:,i]
    
    df.to_csv(args.output)

description = r'''
'''

if __name__ == '__main__':

    from   genolearn.logger import print_dict
    from   genolearn.utils      import check_config

    import argparse

    parser = argparse.ArgumentParser(description = description)

    parser.add_argument('output')
    parser.add_argument('model')
    parser.add_argument('data_config')
    parser.add_argument('values', nargs = '+')
    parser.add_argument('-fs', '--feature-selection', default = None)
    parser.add_argument('-K' , '--nfeatures', default = None)
    parser.add_argument('-k' , '--key', default = None)
    parser.add_argument('-a', '--ascending', default = False, action = 'store_true')

    args   = parser.parse_args()

    params = dict(args._get_kwargs())

    params['data_config'] = check_config(params['data_config'])

    print_dict('executing "genolearn.evaluate.py" with parameters', params)

    main(args)
    