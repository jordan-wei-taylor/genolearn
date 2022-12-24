def train(output_dir, meta, model_config, feature_selection, num_features, binary, min_count, target_subset, metric, aggregate_func):

    from genolearn.utils                 import create_log
    from genolearn.models.classification import get_model
    from genolearn.models                import grid_predictions
    from genolearn.dataloader            import DataLoader
    from genolearn.logger                import msg, Writing, Waiting

    import warnings
    import numpy as np
    import os
    import json
    import pickle


    os.makedirs(output_dir)

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    with open(os.path.join('model', model_config)) as file:
        model_config = json.load(file)
        model        = model_config.pop('model')

    kwargs     = {key : val for key, val in model_config.items() if isinstance(val, list)}
    common     = {key : val for key, val in model_config.items() if key not in kwargs}

    dataloader = DataLoader(meta)
    selection  = dataloader.load_feature_selection(feature_selection).argsort()
    Model      = get_model(model)
    dtype      = bool if binary else None

    outputs, params, X, Y = grid_predictions(dataloader, Model, selection, num_features, dtype, common, min_count, target_subset, metric, aggregate_func, **kwargs)
    
    model, predict, *probs = outputs.pop('best')

    target  = outputs['target']

    npz     = os.path.join(output_dir, 'results.npz')
    pkl     = os.path.join(output_dir, 'model.pickle')
    csv     = os.path.join(output_dir, 'predictions.csv')
    js      = os.path.join(output_dir, 'params.json')
    enc     = os.path.join(output_dir, 'encoding.json')

    dump    = np.c_[outputs['identifiers'], np.array([target, predict]).T]
    headers = ['identifier', 'target', 'predict']

    if probs:
        for i, label in enumerate(dataloader._encoder):
            dump = np.c_[dump, probs[0][:,i].astype(str)]
            headers.append(f'P({label})')

    dump = ','.join(headers) + '\n' + '\n'.join(','.join(row) for row in dump)

    with open(csv, 'w') as f:
        f.write(dump)

    with Writing(npz, inline = True):
        np.savez_compressed(npz, **outputs)

    with Writing(pkl, inline = True):
        with open(pkl, 'wb') as f:
            pickle.dump(model, f)

    with Writing(js, inline = True):
        with open(js, 'w') as f:
            f.write(json.dumps(params, indent = 4))

    with open(enc, 'w') as f:
        f.write(json.dumps(dataloader._encoder, indent = 4))

    num_features = params.pop('num_features')

    with Waiting('fitting', 'fitted', 'full model'):
        full_model = Model(**params).fit(X[:,:num_features], Y)

    path = os.path.join(output_dir, 'full-model.pickle')
    with Writing(path, inline = True):
        with open(path, 'wb') as f:
            pickle.dump(full_model, f)

    create_log('train', output_dir)

    msg(f'executed "genolearn train"')
