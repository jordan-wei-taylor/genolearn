def main(args):
    
    from   genolearn.dataloader import DataLoader
    from   genolearn.models     import grid_predictions
    from   genolearn.logger     import msg, print_dict, Waiting
    from   genolearn.utils      import create_log, subdir

    from   sklearn.linear_model import LogisticRegression
    from   sklearn.ensemble     import RandomForestClassifier

    import warnings
    import numpy as np
    import os

    base   = os.path.dirname(__file__)
    py     = os.path.basename(__file__)
    script = py[:py.index('.')]
    sub    = subdir(base, script)

    params = dict(args._get_kwargs())
    print_dict(f'executing {py} with parameters:', params)

    os.makedirs(base, exist_ok = True)
    os.makedirs(os.path.join(base, sub))

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    dataloader = DataLoader(args.path, args.meta_path, args.identifier, args.target, args.group)

    with Waiting('loading', 'loaded', 'fisher scores', inline = True):
        orders = dataloader.load_feature_selection('fisher-score.npz').rank()
        order  = orders[str(min())]

    train  = range(2014, 2019)
    test   = [2019]

    K_rf   = [100, 1000, 10000, 100000, 1000000]
    K_lr   = K_rf[:-1]

    common_rf = dict(n_jobs = -1, class_weight = 'balanced')
    common_lr = dict(n_jobs = -1, class_weight = 'balanced', solver = 'saga')

    params_rf = dict(max_depth = range(5, 101, 5), random_state = range(10))
    params_lr = dict(C = [1e-2, 1, 1e2], random_state = range(10))

    rf = ['random-forest.npz', RandomForestClassifier, K_rf, common_rf, params_rf]
    lr = ['logistic-regression.npz', LogisticRegression, K_lr, common_lr, params_lr]

    for file, model, K, common, params in [rf, lr]:

        msg(f'computing contents for {file}')
        hats, times = grid_predictions(dataloader, train, test, model, K, order, common, **params, min_count = args.min_count)

        
        with Waiting('generating', 'generated', file):
            np.savez(os.path.join(base, 'script-output', sub, file), hats = hats, times = times, K = K, **params)

    create_log('script-output', f'{script}-log.txt')

    msg(f'executed {py}')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('compares Random Forest and Logistic Regression models')

    parser.add_argument('path')
    parser.add_argument('meta_path')
    parser.add_argument('identifier')
    parser.add_argument('target')
    parser.add_argument('-group', default = 'Year')
    parser.add_argument('-min_count', default = 15)

    args = parser.parse_args()

    main(args)
