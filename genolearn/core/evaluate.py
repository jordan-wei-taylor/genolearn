def evaluate(data_config, feature_selection, values, encoder, num_features, binary, output):

    from   genolearn.logger import msg

    from   genolearn.dataloader import DataLoader

    import joblib
    import pandas as pd

    import os

    dtype = bool if binary else None

    for model, pre in zip(['model.pickle', 'full-model.pickle'], ['', 'full-']):
        model       = joblib.load(os.path.join('..', model))
        dataloader  = DataLoader(**data_config)

        features = dataloader.load_feature_selection(feature_selection).argsort()[:num_features]

        X   = dataloader.load_X(*values, features = features, dtype = dtype)

        dataloader._encoder = encoder

        df  = pd.DataFrame(index = dataloader._check_identifiers(values))
        df.index.name = 'identifier'

        df['hat'] = dataloader.decode(model.predict(X))
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)
            for i, name in enumerate(encoder):
                df[f'P({name})'] = prob[:,i]
        
        df.to_csv(f'{pre}{output}')

    msg('executed "genolearn evaluate"')
