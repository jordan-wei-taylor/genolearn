def _feature_selection(dataloader, init, inner_loop, outer_loop, values, force_dense = False, force_sparse = False):
    ret, args, kwargs = init(dataloader)
    for value in values:
        for i, (x, label) in enumerate(dataloader.generator(value, force_dense = force_dense, force_sparse = force_sparse), 1):
            inner_loop(ret, i, x, label, value, *args, **kwargs)
        outer_loop(ret, i, value, *args, **kwargs)
    return ret
