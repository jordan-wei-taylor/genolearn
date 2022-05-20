Quickstart Guide
################

In this Quickstart guide, we assume that the user has preprocessed their dataset according to the :ref:`Preprocessing <Preprocessing>` section with their meta data, ``meta-data.csv``, and the resulting data subdirectory ``data`` in the same directory. Since execution of the preprocessing step can be lengthy we provide an example dataset via GitHub.

.. code-block:: sh

    git clone https://github.com/jordan-wei-taylor/genolearn-demo-data.git

The dataset contains data relating to E. Coli O157 with 2,784 different strains, each with a count vector of over 12 million :math:`k`-mers, and an associated region of origin in the meta data file for years 2014 to 2019. We can note this is a large dataset so in order to run any machine learning models on this dataset we will need a means of selecting which genome sequences are of interest. To do so we can use the Fisher Scores for each genome sequence (see :ref:`FeatureSelection`). Lets compute the Fisher Scores for each genome sequence with the below command

.. code-block:: sh

    python3 -m genolearn.feature_selection fisher-scores.npz genolearn-demo-data/data genolearn-demo-data/meta-data.csv Accession Regions 2018 2017 2016 2015 2014 -group Year

The above command computes the fisher scores for all genome sequence counts from years 2018 - 2018, then 2017 - 2018, ..., 2014 - 2018. This is to simulate the effect of collecting more data i.e. in scenario one, we only have a single year's worth of data, and in the final scenario, we have 5 years worth of data.

Upon execution of the above command we should the path ``genolearn-demo-data/feature-selection/fisher-scores.npz`` should exist. Before training a machine learning model, we may wish to specify both a ``data_config`` and ``model_config`` files. Here is an example of the contents for both:

.. code-block:: text
    :caption: data_config.json

    {
        "path": "genolearn-demo-data",
        "meta_path": "genolearn-demo-data/meta-data.csv",
        "identifier": "Accession",
        "target": "Region",
        "group": "Year"
    }

This specifies that the path to all of the outputs from the :ref:`Preprocessing` stage is ``genolearn-demo-data``, similarly the meta data is present in the same directory. We further state that the identifier for each strain is the ``Accession`` column within the meta data file, and the target values of interest are located within the ``Region`` column. Further we state that strains can be grouped by the ``Year`` column.

.. code-block:: text
    :caption: model_config.json

    {
        "n_jobs": -1,
        "class_weight": "balanced",
        "max_depth": [
            10,
            30,
            50
        ],
        "random_state": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9
        ]
    }

In this model config file, we specify any key word arguments to sklearn's `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_. If we would like to try a range of values we can do so by specifying a list of values. In the above example we are training models with ``max_depth`` values of 10, 30, and 50 as well as 10 different ``RandomState`` values.

Now lets run a training script where we train on all strains collected in 2018 and compute predictions on strains collected in 2019 using the most important 10000 genome sequences.

.. code-block:: sh

    python3 -m genolearn.train output RandomForestClassifier data_config.json rf_config.json -train 2018 -test 2019 -K 10000 -order fisher-score.npz -order_key 2018

The above generates an ``output/results.npz`` file which works similarly to a dictionary. In Python we can access the information through indexing string values.

.. code-block:: python

    import numpy as np

    npz = np.load('output.npz')

    identifiers = npz['identifiers'] # shape (n,)
    predictions = npz['predict']     # shape (1, 3, 10, n)
    times       = npz['times']       # shape (1, 3, 10, 2)

where :math:`n` is the number of valid testing strains present in 2019 i.e. only the strains in 2019 where the associated target region was present in 2018. This ignores strains in 2019 where the associated target region does not appear in 2018 as there are no training examples to learn from.




