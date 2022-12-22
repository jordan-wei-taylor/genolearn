def train(output_dir, meta, model_config, feature_selection, num_features, min_count, target_subset, metric, aggregate_func):

    import os
    import json

    command = 'genolearn train'

    from genolearn.utils                 import create_log
    from genolearn.models.classification import get_model
    from genolearn.models                import grid_predictions
    from genolearn.dataloader            import DataLoader
    from genolearn.logger                import msg, Writing

    import warnings
    import numpy as np
    import os
    import json
    import pickle


    os.makedirs(output_dir)

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


    # data_config, model_config = map(check_config, (data_config, model_config))

    # with open(model_config) as f:
    #     model_config = json.load(model_config)


    # print_dict('executing "train.py" with parameters', params)

    with open(os.path.join('model', model_config)) as file:
        model_config = json.load(file)
        model        = model_config.pop('model')

    kwargs     = {key : val for key, val in model_config.items() if isinstance(val, list)}
    common     = {key : val for key, val in model_config.items() if key not in kwargs}

    dataloader = DataLoader(meta)
    selection  = dataloader.load_feature_selection(feature_selection).argsort()

    Model   = get_model(model)
    
    outputs, params = grid_predictions(dataloader, Model, selection, num_features, common, min_count, target_subset, metric, aggregate_func, **kwargs)
        
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

    create_log('train', output_dir)

    msg(f'executed "{command}"')
