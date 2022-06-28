Train
#####

Suppose we are interested in training a Machine Learning model on our preprocessed dataset. Given a Machine Learning model, we may wish to:

+ vary it's hyperparameters
+ evaluate a goodness-of-fit
+ save the best model for later evaluation

As an example, suppose we want to train a Random Forest model over the period 2014 - 2018 and evaluate it on 2019. We would like to vary the number of features to use as per the Fisher Scores to be 100, 1,000, ..., 1000,000, and we would like to compute the goodness of fit of our model according to the ``recall`` metric. Further we would like to try out different hyperparameter tunings of our Random Forest by varying the ``max_depth`` and ``random_state``. Lets create the required model and data config files.

.. code-block:: bash

    python3 -m genolearn.utils.make_config model_config.json '{"n_jobs" : -1, "class_weight" : "balanced", "max_depth" : range(5, 105, 5), "random_state" : range(10)}'
    python3 -m genolearn.utils.make_config data_config.json '{"path" : "data", "meta_path" : "raw-data/meta-data.csv", "identifier" : "Accession", "target" : "Region", "group" : "Year"}'

The above should generate two files in your current directory

.. code-block:: text

    {
        "n_jobs": -1,
        "class_weight": "balanced",
        "max_depth": [
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            100
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


.. code-block:: text

    {
        "path": "data",
        "meta_path": "raw-data/meta-data.csv",
        "identifier": "Accession",
        "target": "Region",
        "group": "Year"
    }


Training the ``RandomForestClassifier`` varying the earlier specified hyperparameters is done so by executing

.. code-block:: bash

    output="output"
    model="RandomForestClassifier"
    train="2018 2017 2016 2015 2014"
    test="2019"
    K="100 1000 10000 100000 1000000"
    order="fisher-scores.npz"
    key="2014"
    metric="recall"

    python3 -m genolearn.train $output $model data_config.json model_config.json -train $train -test $test -K $K -order $order -order_key $key -metric $metric

The above should generate the following directory

.. code-block:: text

    output
    ├── log.txt
    ├── model.pickle
    ├── params.json
    ├── predictions.csv
    └── results.npz
    

where ``log.txt``, is a text file containing information like what parameters were used when calling ``genolearn.train``, time taken to complete command, and RAM usage. ``model.pickle`` is the saved best model, ``param.json`` is a json file containing the best parameter setting found when a grid search was executed as per the specifications of the ``model_config.json`` file, and ``predictions.csv`` are results on the 2019 dataset. This is chosen to be the best model according the recall metric when training on data gathered between 2014-2018 and evaluated on 2019 where we choose genome sequences according to the Fisher Scores based on the 2014-2018 dataset.
