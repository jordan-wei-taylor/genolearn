Tutorial
################

In this Tutorial, we are interested in using :math:`k`-mer counts to predict where an E. Coli O157 strain geographically originated from. We imagine ourselves having collected data from years 2014-2018 and find ourselves in 2019. We want to preprocess the 2014-2018 dataset, perform feature selection, and train machine learning models on this data. We then collect data in 2019 and wish to see how accurate our machine learning models are for the new data. For demo purposes, the data has been truncated to contain only the first 100,000 :math:`k`-mers which means reduces execution times at the cost of model performance.

We start by cloning an E. Coli O157 dataset which consists kmer count files (split into parts) and location region as the meta data. For the complete dataset clone my https://github.com/jordan-wei-taylor/e-coli-o157-data.git and execute ``./merge.sh`` after executing ``cd genolearn-e-coli``.

.. code-block:: bash

    git clone https://github.com/jordan-wei-taylor/genolearn-tutorial.git

Upon cloning the repository, ``cd`` into the repository.

.. code-block:: bash

    cd genolearn-tutorial

Once the above has been executed, your directory tree should look like the following:

.. code-block:: text

    genolearn-tutorial
        ├── data
        │   ├── 14-18.txt.gz
        │   ├── 19.txt.gz
        │   └── metadata.csv
        └── README.md

Lets install ``GenoLearn`` on a virtual python environment.

.. code-block:: bash

    python3 -m venv env
    source env/bin/activate
    pip install git+https://github.com/jordan-wei-taylor/genolearn.git

Now that we have the most recent installation of ``GenoLearn``, lets setup ``GenoLearn`` for our current directory.

.. code-block:: bash

    genolearn-setup


.. rubric:: Prompted Information
.. code-block:: text

    data directory : data
    metadata csv   : metadata.csv

With setup complete, we can just run the command ``genolearn``

.. code-block:: text

    GenoLearn ({VERSION}) Command Line Interface

    GenoLearn is designed to enable researchers to perform Machine Learning on their genome
    sequence data such as fsm-lite or unitig files.

    See https://genolearn.readthedocs.io for documentation.

    Working directory: ~/genolearn-demo 

    1.  exit                            exits GenoLearn

    2.  print                           prints various GenoLearn generated files
    3.  preprocess                      preprocess data into an easier format for file reading


As the user executes the commands available, new commands are made available. Lets start by preprocessing the 2014-2018 dataset by selecting ``preprocess`` followed by ``sequence`` upon executing ``genolearn``.

.. note::

    For the rest of the tutorial, if it is stated to run a ``command``, we mean you should execute ``genolearn`` then select the ``command`` number from the numbered menu as shown above.
    

.. rubric:: Prompted Information
.. code-block:: text

    sequence data      : 14-18.txt.gz
    batch-size   [None]:
    n-processes  [None]:
    sparse       [True]:
    dense        [True]:
    verbose    [250000]:
    max features [None]:


Following the execution of the above, which should take up to a few minutes, your directory tree should look like

.. code-block:: text
    :emphasize-lines: 6-10

    genolearn-tutorial
        ├── data
        │   ├── 14-18.txt.gz
        │   ├── 19.txt.gz
        │   └── metadata.csv
        ├── preprocess
        │   ├── array   [2460 .npz files]
        │   ├── features.txt.gz
        │   ├── info.json
        │   └── preprocess.log
        └── README.md

Imagine, at a later point in time, we had access to the 2019 dataset. We can combine the preprocessing if this data with our previous data with the ``preprocess combine`` command.


.. rubric:: Prompted Information
.. code-block:: text

    sequence data      : 19.txt.gz
    batch-size   [None]:
    n-processes  [None]:


The directory tree following the above execution should be

.. code-block:: text
    :emphasize-lines: 7, 8

    genolearn-tutorial
    ├── data
    │   ├── 14-18.txt.gz
    │   ├── 19.txt.gz
    │   └── metadata.csv
    ├── preprocess
    │   ├── combine.log
    │   ├── array   [2875 .npz files]
    │   ├── features.txt.gz
    │   ├── info.json
    │   ├── meta.json
    │   └── preprocess.log
    └── README.md


Now we need a means of splitting our data into a *train* and *validation* datasets. The *train* set is used to fit our models whilst the *validation* set is used to check model performance which then helps us decide which of the potentially many models to choose from. For this tutorial, since we want to train on the 2014 - 2018 dataset and then evaluate against the 2019 dataset we execute ``preprocess meta`` with the following parameters

.. rubric:: Prompted Information
.. code-block:: text

    output        [default]: yearly
    identifier             : Accession
    target                 : Region
    group            [None]: Year
    train group values*    : 2014, 2015, 2016, 2017, 2018
    val group values*      : 2019


If you do not have something similar to the ``Year`` column in your metadata, it is recommended to use the default values provided.


.. rubric:: Prompted Information
.. code-block:: text

    output        [default]: 
    identifier             : Accession
    target                 : Region
    group            [None]: 
    proportion train [0.75]: 


This randomly assigns 75\% of the data as the *train* dataset and the rest as your *test* dataset but the user is free to change the proportion that should be in the *train* dataset by entering a sensible proportion value. The directory tree should be

.. code-block:: text
    :emphasize-lines: 6-8

    genolearn-tutorial
    ├── data
    │   ├── 14-18.txt.gz
    │   ├── 19.txt.gz
    │   └── metadata.csv
    ├── meta
    │   ├── default
    │   └── yearly
    ├── preprocess
    │   ├── combine.log
    │   ├── array   [2875 .npz files]
    │   ├── features.txt.gz
    │   ├── info.json
    │   ├── meta.json
    │   └── preprocess.log
    └── README.md


At this point we can analyse the metadata by executing ``print meta analyse``. If we choose our ``yearly`` metadata, it prints

.. code-block:: text

                      |   2014 |   2015 |   2016 |   2017 |   2018 |   2019 |  Train |    Val | Global
    ------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------
    Asia              |      0 |      4 |     16 |      9 |     18 |     10 |     47 |     10 |     57
    Australasia       |      0 |      0 |      1 |      0 |      3 |      2 |      4 |      2 |      6
    C. America        |      0 |      2 |      6 |      6 |     10 |      5 |     24 |      5 |     29
    C. Europe         |      0 |      0 |     26 |     15 |      8 |     13 |     49 |     13 |     62
    M. East           |      3 |     11 |     33 |     23 |     42 |     45 |    112 |     45 |    157
    N. Africa         |      0 |     15 |      7 |     26 |     24 |     19 |     72 |     19 |     91
    N. America        |      1 |      1 |      3 |      3 |      3 |      1 |     11 |      1 |     12
    N. Europe         |      0 |      1 |      2 |      7 |      2 |      6 |     12 |      6 |     18
    S. America        |      0 |      0 |      1 |      2 |      0 |      0 |      3 |      0 |      3
    S. Europe         |      5 |     13 |     62 |     59 |     54 |     44 |    193 |     44 |    237
    Subsaharan Africa |      3 |      2 |      7 |      7 |      6 |      3 |     25 |      3 |     28
    UK                |    133 |    135 |    765 |    406 |    468 |    267 |  1,907 |    267 |  2,174
    Total             |    145 |    184 |    929 |    563 |    638 |    415 |  2,459 |    415 |  2,874

    suggested target subset: Asia, C. America, C. Europe, M. East, N. Africa, N. America, N. Europe, S. Europe, Subsaharan Africa, UK


From the above, we can see the target region distribution over each year as well as the *train* and *validation* datasets. Additionally, it prints the target regions that have a count of at least 10 (by default) to ensure our later models have enough examples to learn from (the number you choose is somewhat subjective).

If we instead, choose to print out the ``default`` metadata, it prints

.. code-block:: text

                      |  Train |    Val | Global
    ------------------+--------+--------+--------
    Asia              |     43 |     14 |     57
    Australasia       |      5 |      1 |      6
    C. America        |     22 |      7 |     29
    C. Europe         |     38 |     24 |     62
    M. East           |    111 |     46 |    157
    N. Africa         |     73 |     18 |     91
    N. America        |      9 |      3 |     12
    N. Europe         |     14 |      4 |     18
    S. America        |      2 |      1 |      3
    S. Europe         |    187 |     50 |    237
    Subsaharan Africa |     24 |      4 |     28
    UK                |  1,628 |    546 |  2,174
    Total             |  2,156 |    718 |  2,874

    suggested target subset: Asia, C. America, C. Europe, M. East, N. Africa, N. Europe, S. Europe, Subsaharan Africa, UK


.. note::

    Numbers suggested target subset may vary due to the randomness.


At this point, we can now use Fisher Scores to compute which genome sequences to take forward for modelling purposes (see `Feature Selection <https://genolearn.readthedocs.io/en/stable/usage/feature-selection.html>`_ for more details). Lets compute the Fisher Scores for each genome sequence with the ``feature-selection`` command using the ``yearly`` metadata.

.. rubric:: Prompted Information
.. code-block:: text

    metadata            : yearly
    name [yearly-fisher]: 


Upon execution of the above command, the directory tree should be now

.. code-block:: text
    :emphasize-lines: 6-8

    genolearn-tutorial
    ├── data
    │   ├── 14-18.txt.gz
    │   ├── 19.txt.gz
    │   └── metadata.csv
    ├── feature-selection
    │   ├── yearly-fisher
    │   └── yearly-fisher.log
    ├── meta
    │   ├── default
    │   └── yearly
    ├── preprocess
    │   ├── combine.log
    │   ├── array   [2875 .npz files]
    │   ├── features.txt.gz
    │   ├── info.json
    │   ├── meta.json
    │   └── preprocess.log
    └── README.md


The above command computes the fisher scores for all genome sequence counts that have been assigned as *train*. 

Now that we have preprocessed all of our data and computed the Fisher Scores, we can look to see how our Machine Learning models perform on our dataset. Before training a machine learning model, we will need define a ``model config``. Below is an example Random Forest config with varying ``max_depth`` and ``random_states`` after executing the ``model-config`` command

.. rubric:: Prompted Information
.. code-block:: text

    config_name [random-forest]                             : 
    n_estimators [100]                                      : 
    criterion {gini, entropy, log_loss} [gini]              : 
    max_depth [None]                                        : range(10, 51, 10)
    min_samples_split [2]                                   : 
    min_samples_leaf [1]                                    : 
    min_weight_fraction_leaf [0.0]                          : 
    max_features {sqrt, log2, None} [sqrt]                  : 
    max_leaf_nodes [None]                                   : 
    min_impurity_decrease [0.0]                             : 
    bootstrap [True]                                        : 
    oob_score [False]                                       : 
    n_jobs [-1]                                             : 
    random_state [None]                                     : 0, 1, 2
    class_weight {balanced, balanced_subsample, None} [None]: balanced


.. note::

    Note that you can enter a range of integers either manually like the ``random_state`` entry or use Python's ``range`` function.


The above results in the following directory tree.

.. code-block:: text
    :emphasize-lines: 12, 13

    genolearn-tutorial
    ├── data
    │   ├── 14-18.txt.gz
    │   ├── 19.txt.gz
    │   └── metadata.csv
    ├── feature-selection
    │   ├── yearly-fisher
    │   └── yearly-fisher.log
    ├── meta
    │   ├── default
    │   └── yearly
    ├── model
    │   └── random-forest
    ├── preprocess
    │   ├── combine.log
    │   ├── array   [2875 .npz files]
    │   ├── features.txt.gz
    │   ├── info.json
    │   ├── meta.json
    │   └── preprocess.log
    └── README.md


Using the ``print`` command, the ``random-forest`` file contents are

.. rubric:: random-forest
.. code-block:: text

    {
        model: RandomForestClassifier,
        n_estimators: 100,
        criterion: gini,
        max_depth: [
            10,
            20,
            30,
            40,
            50
        ],
        min_samples_split: 2,
        min_samples_leaf: 1,
        min_weight_fraction_leaf: 0.0,
        max_features: sqrt,
        max_leaf_nodes: null,
        min_impurity_decrease: 0.0,
        bootstrap: true,
        oob_score: false,
        n_jobs: -1,
        random_state: [
            0,
            1,
            2
        ],
        class_weight: balanced
    }


With these files now created, we can proceed to the ``train`` command. Lets assume we want to search for models that use the top 1,000, 10,000 and then the full 100,000 features as computed by the earlier ``Fisher Scores`` and further assume we want to only train / evaluate on regions where we have at least 10 examples to learn from.

.. rubric:: Prompted Information
.. code-block:: text

    output_dir [yearly-fisher-random-forest]            : 
    num_features* [1000]                                : 1000, 10000, 100000
    binary [False]                                      : 
    min_count [0]                                       : 10
    target_subset [None]                                : 
    metric [f1_score]                                   : 
    aggregate_func {mean, weighted_mean} [weighted_mean]:


The above results in the following directory tree

.. code-block:: text
    :emphasize-lines: 22-

    genolearn-tutorial
    ├── data
    │   ├── 14-18.txt.gz
    │   ├── 19.txt.gz
    │   └── metadata.csv
    ├── feature-selection
    │   ├── yearly-fisher
    │   └── yearly-fisher.log
    ├── meta
    │   ├── default
    │   └── yearly
    ├── model
    │   └── random-forest
    ├── preprocess
    │   ├── combine.log
    │   ├── array   [2875 .npz files]
    │   ├── features.txt.gz
    │   ├── info.json
    │   ├── meta.json
    │   └── preprocess.log
    ├── README.md
    └── train
        └── yearly-fisher
            ├── full-model.pickle
            ├── encoding.json
            ├── model.pickle
            ├── params.json
            ├── predictions.csv
            ├── results.npz
            └── train.log


For a deeper insight on the results, you will need to run in Python

.. code-block:: python

    import numpy as np

    npz = np.load('path/to/results.npz')

    identifiers = npz['identifiers'] # shape (n,)
    predictions = npz['predict']     # shape (3, 5, 3, n) (3 num_features, 5 max_depth, and 3 random_state parameters for n predictions)
    times       = npz['times']       # shape (3, 5, 3, 2) (last dimension measures training and predicting times in seconds)

where :math:`n` is the number of validation strains present in 2019 i.e. only the strains in 2019 where the associated target region was present in 2018. This ignores strains in 2019 where the associated target region does not appear in 2018 as there are no training examples to learn from.

Feature importances can be obtained by the ``feature-importance`` command which results in the following directory tree

.. code-block:: text
    :emphasize-lines: 26-30

    genolearn-tutorial
    ├── data
    │   ├── 14-18.txt.gz
    │   ├── 19.txt.gz
    │   └── metadata.csv
    ├── feature-selection
    │   ├── yearly-fisher
    │   └── yearly-fisher.log
    ├── meta
    │   ├── default
    │   └── yearly
    ├── model
    │   └── random-forest
    ├── preprocess
    │   ├── combine.log
    │   ├── array   [2875 .npz files]
    │   ├── features.txt.gz
    │   ├── info.json
    │   ├── meta.json
    │   └── preprocess.log
    ├── README.md
    └── train
        └── yearly-fisher
            ├── full-model.pickle
            ├── encoding.json
            ├── importance
            │   ├── full-importance.npz
            │   ├── full-importance-rank.csv
            │   ├── importance.npz
            │   └── importance-rank.csv
            ├── model.pickle
            ├── params.json
            ├── predictions.csv
            ├── results.npz
            └── train.log

The ``importance-rank.csv`` contains a single column of genome sequences which have been ranked from most important to least by the trained model's logic based on the 2014 - 2018 dataset. See `Feature Importance <https://genolearn.readthedocs.io/en/stable/usage/feature-importance.html>`_ for further details on interpretation of the ``importance.npz`` file.

Finally, if new data is obtained but with no corresponding target labels (i.e. only the genome sequence counts are obtained), then after first executing ``preprocess combine`` on the new data, we can evaluate on the new ``unlabelled`` data points.

.. rubric:: Prompted Information
.. code-block:: text

    output filename                                               : unlabelled
    group values* {2014, 2015, 2016, 2017, 2018, 2019, unlabelled}: unlabelled

which results in the following directory tree

.. code-block:: text
    :emphasize-lines: 26-28

    genolearn-tutorial
    ├── data
    │   ├── 14-18.txt.gz
    │   ├── 19.txt.gz
    │   └── metadata.csv
    ├── feature-selection
    │   ├── yearly-fisher
    │   └── yearly-fisher.log
    ├── meta
    │   ├── default
    │   └── yearly
    ├── model
    │   └── random-forest
    ├── preprocess
    │   ├── combine.log
    │   ├── array   [2875 .npz files]
    │   ├── features.txt.gz
    │   ├── info.json
    │   ├── meta.json
    │   └── preprocess.log
    ├── README.md
    └── train
        └── yearly-fisher
            ├── full-model.pickle
            ├── encoding.json
            ├── evaluate
            │   ├── full-unlabelled.npz
            │   └── unlabelled.csv
            ├── importance
            │   ├── full-importance.npz
            │   ├── full-importance-rank.csv
            │   ├── importance.npz
            │   └── importance-rank.csv
            ├── model.pickle
            ├── params.json
            ├── predictions.csv
            ├── results.npz
            └── train.log

Since for our dataset, we had 1 example not labelled (i.e. it does not appread in the metadata but appears in one of preprocessed .gz files), ``unlabelled.csv`` looks like

.. code-block:: text

     identifier hat   P(Asia)  P(C. America)  P(C. Europe)  P(M. East)  P(N. Africa)  P(N. America)  P(N. Europe)  P(S. Europe)  P(Subsaharan Africa)     P(UK)
    SRR10001272  UK  0.092434       0.052168      0.125289    0.067996      0.106553       0.134874      0.098422      0.136982              0.026605  0.158678


