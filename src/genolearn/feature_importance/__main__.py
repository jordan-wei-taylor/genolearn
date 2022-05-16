if __name__ == '__main__':

    from   genolearn.dataloader         import DataLoader
    from   genolearn.feature_importance import FeatureImportance
    from   genolearn.logger             import Writing
    
    import argparse
    import pickle

    import numpy as np


    parser = argparse.ArgumentParser('feature_importance script')

    parser.add_argument('path')
    parser.add_argument('feature_selection')
    parser.add_argument('key')
    parser.add_argument('model')
    parser.add_argument('outpath')

    args = parser.parse_args()

    dataloader = DataLoader(args.path, None, None, None)

    selection  = dataloader.load_feature_selection(args.feature_selection)

    features   = dataloader.features(selection[args.key])

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    importance = FeatureImportance(model)

    scores     = importance.feature_scores
    ranks      = importance.feature_ranks()

    with Writing(args.outpath):
        np.savez(args.outpath, features = features[:len(ranks)], ranks = ranks, scores = scores)

