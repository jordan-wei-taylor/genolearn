.. _DataLoader:

DataLoader
##########

``genolearn`` provides the user with a ``DataLoader`` class to load the data with ease given a path to a preprocessed directory output by running the previous `preprocessing <../preprocessing.html>`__ step.

In the following examples it is assumed the user had a directory similar to

.. code-block:: text

    .
    └── raw-data
        ├── 14-19-kmers.txt.gz
        └── meta-data.csv

By running the preprocessing script

.. code-block:: text

    >>> python -m genolearn data raw-data/14-19-kmers.txt.gz --batch_size 512

the directory should have been modified to contain the following subdirectory.

.. code-block:: text

    .
    ├── data
    │   ├── dense
    │   │   └── *.npz
    │   ├── feature-selection
    │   ├── features.txt
    │   ├── log.txt
    │   ├── meta.json
    │   └── sparse
    │       └── *.npz
    └── raw-data
        ├── 14-19-kmers.txt.gz
        └── meta-data.csv

Basic Example
=========================

.. code-block:: python

    from genolearn.dataloader import DataLoader

    # instantiate DataLoader
    dataloader = DataLoader('data', 'raw-data/meta-data.csv', 'id', 'region', group = 'year')

    # loads a single count array with associated identifier 'a' i.e. 'data/dense/a.npz' exists
    x = dataloader.load_X('a')

    # loads a 2D count array with associated identifiers belonging to year = 2018
    X = dataloader.load_X(2018)

    # similar to the above but with associated identifiers belonging to years {2017, 2018} and with the associated target region values
    X, Y = dataloader.load(2017, 2018)

Advanced Example
============================

Suppose we have computed the `Fisher Scores <../feature-selection/fisher-score-example.html>`__ by first executing

.. code-block:: text

    # generates fisher scores for periods 2018-2018 and 2017-2018
    python -m genolearn.feature_selection fisher-scores.npz data raw-data/meta-data.csv id region 2018 2017 -group year

which modifies our directory to contain an additional file in the feature-selection subdirectory as shown below.

.. code-block:: text

    .
    ├── data
    │   ├── dense
    │   │   └── *.npz
    │   ├── feature-selection
    │   │   └── fisher-score.npz
    │   ├── features.txt
    │   ├── log.txt
    │   ├── meta.json
    │   └── sparse
    │       └── *.npz
    └── raw-data
        ├── 14-19-kmers.txt.gz
        └── meta-data.csv

We can use these scores to select genome sequences with the top :math:`k` scores.

.. code-block:: python

    from genolearn.dataloader import DataLoader

    # instantiate DataLoader
    dataloader = DataLoader('data', 'raw-data/meta-data.csv', 'id', 'region', group = 'year')

    # load the Fisher Scores dictionary
    scores     = dataloader.load_feature_selection('fisher-score.npz')

    # ranks the features from highest to lowest
    order      = scores.rank()

    # compute the top 1000 scoring features for the period 2017-2018
    k          = 1000
    features   = order['2017'][:k]

    # load all count arrays belonging to years {2017, 2018} with 1000 columns associated 
    # to the top 1000 scoring Fisher Scores and their associated target region values
    X, Y       = dataloader.load(2017, 2018, features = features)

    # define which years belong to the train and test datasets
    train      = [2017, 2018]
    test       = [2019]

    # define train data to be data points associated with years {2017, 2018} and test data
    # to be those associated with the year 2019 only both focusing on genome sequences that
    # had the top 1000 Fisher Scores for the training period 
    X_train, Y_train, X_test, Y_test = dataloader.load_train_test(train, test, features = features)