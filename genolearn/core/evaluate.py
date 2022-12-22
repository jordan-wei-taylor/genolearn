def evaluate(model, data_config, feature_selection, values, encoder, num_features, output):

    from   genolearn.logger import msg

    from   genolearn.dataloader import DataLoader

    import joblib
    import pandas as pd

    model       = joblib.load(model)
    dataloader  = DataLoader(**data_config)

    features = dataloader.load_feature_selection(feature_selection).argsort()[:num_features]

    X   = dataloader.load_X(*values, features = features)

    dataloader._encoder = encoder

    df  = pd.DataFrame(index = dataloader._check_identifiers(values))
    df.index.name = 'identifier'

    df['hat'] = dataloader.decode(model.predict(X))
    
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)
        for i, name in enumerate(encoder):
            df[f'P({name})'] = prob[:,i]
    
    df.to_csv(output)

    msg('executed "genolearn evaluate"')
