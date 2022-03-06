Data Loader
##########################


.. code-block:: python

    from biolearn.dataloader import DataLoader

    # assume 'data-by-year' is a directory output by the main biolearn package for kmer data

    # control the number of processes for parallelised reading with n_processes
    dataloader = DataLoader('data-by-year', n_processes = 8)

    # obtain both the k-mer counts and associated meta-data for years 2017, 2018, and 2019
    X, Y = dataloader.load(2017, 2018, 2019)

    # same as above but only considering the first 100 k-mers
    X, Y = dataloader.load(2017, 2018, 2019, mask = range(100))

    # same as above but with X to only contain values in {0, 1} i.e. 0 -> 0, 1+ -> 1
    X, Y = dataloader.load(2017, 2018, 2019, mask = range(100), dtype = bool) # same as applying X = X.astype(bool) to the above
    
    # obtains the columns of X (a list of k-mer strings if used with fsm files)
    columns = dataloader.columns

    # obtains the row identifiers for both X and Y (a list of SRR ids if used with fsm files)
    identifiers = dataloader.identifiers

    