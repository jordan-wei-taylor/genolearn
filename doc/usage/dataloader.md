.. _DataLoader:

Data Loader
##########################


.. code-block:: python

    from genolearn import DataLoader

    # assume 'data' is a directory output by the main genolearn package for k-mer data
    dataloader = DataLoader('data', meta_path = 'raw-data/meta-data.csv', identifier = 'Accession', group = 'Year', target = 'Region')

    # obtain both the k-mer counts and associated meta-data for identifiers 'SRR3530392' and 'SRR3530530'
    X, Y = dataloader.load('SRR3530392', 'SRR3530530')

    # obtain both the k-mer counts and associated meta-data for all identifiers that has the associated year value of 2017, 2018, or 2019
    X, Y = dataloader.load(2017, 2018, 2019)

    # same as above but only considering the first 100 k-mers
    X, Y = dataloader.load(2017, 2018, 2019, features = range(100))

    # same as above but with X to only contain values in {0, 1} i.e. 0 -> 0, 1+ -> 1
    X, Y = dataloader.load(2017, 2018, 2019, features = range(100), dtype = bool) # same as applying X = X.astype(bool) to the above
    
    # obtains the columns of X (a list of k-mer strings if used with fsm files) to the most recent run of loading data
    features = dataloader.features

    # obtains the row identifiers for both X and Y (a list of SRR ids if used with fsm files) to the most recent run of loading data
    identifiers = dataloader.identifiers

    