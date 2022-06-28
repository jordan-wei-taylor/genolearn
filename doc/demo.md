Demo
################

In this Demo, we imagine ourselves having collected data from years 2014-2018 and find ourselves in 2019. We want to preprocess the 2014-2018 dataset, perform feature selection, and train machine learning models on this data. We then collect data in 2019 and wish to see how accurate our machine learning models are for the new data.

We start by cloning an E. Coli O157 dataset which consists kmer count files (split into parts) and location region as the meta data. We have split the data into parts as per the maximum file size guidelines listed by GitHub (100 MB).

.. code-block:: bash

    git clone https://github.com/jordan-wei-taylor/e-coli-o157-data.git

Upon cloning the repository, ``cd`` into the repository and execute the ``merge.sh`` script to merge all part files.

.. code-block:: bash

    cd e-coli-o157-data
    ./merge.sh

Once the above has been executed there should be three files within the directory ``raw-data``.

.. code-block:: text

    e-coli-o157-data
    ├── merge.sh
    └── raw-data
        ├── meta-data.csv
        ├── STEC_14-18_fsm_kmers.txt.gz
        └── STEC_19_fsm_kmers.txt.gz

Lets install ``GenoLearn`` on a virtual python environment.

.. code-block:: bash

    python3 -m venv env
    source env/bin/activate
    pip install git+https://github.com/jordan-wei-taylor/genolearn.git

Now that we have the most recent installation of ``GenoLearn``, lets preprocess the data.

.. code-block:: bash

    output="data"
    input="raw-data/STEC_14-18_fsm_kmers.txt.gz"

    python3 -m genolearn.preprocess $output $input

    # execute this instead for quicker preprocessing but a higher diskspace requirement
    # ulimit -n 8000 # remove maximum number of files open at a time
    # python3 -m genolearn.preprocess $output $input -bs -1

The dataset contains data relating to E. Coli O157 with 2,460 different strains, each with a count vector of over 12 million :math:`k`-mers, and an associated region of origin in the meta data file for years 2014 to 2018. We can note this is a large dataset so in order to run any machine learning models on this dataset we will need a means of selecting which genome sequences are of interest. To do so we can use the Fisher Scores for each genome sequence (see :ref:`FeatureSelection`). Lets compute the Fisher Scores for each genome sequence with the below command

.. code-block:: bash

    output="fisher-scores.npz"
    dataset="data"
    meta="raw-data/meta-data.csv"
    identifier="Acccession"
    target="Region"
    group="Year"
    group_vals="2018 2017 2016 2015 2014"

    python3 -m genolearn.feature_selection $output $dataset $meta $identifier $target $group_vals -group $group


Upon execution of the above command, we should the path ``data/feature-selection/fisher-scores.npz`` should exist.  The above command computes the fisher scores for all genome sequence counts from years 2018 - 2018, then 2017 - 2018, ..., 2014 - 2018. This is to simulate the effect of collecting more data i.e. in scenario one, we only have a single year's worth of data, and in the final scenario, we have 5 years worth of data.

Now lets add to our preprocessed data, newly collected data from 2019.

.. code-block:: bash

    output="data"
    input="raw-data/STEC_19_fsm_kmers.txt.gz"
    
    python3 -m genolearn.combine $output $input

    # execute this instead for quicker preprocessing but a higher diskspace requirement
    # python3 -m genolearn.combine $output $input -bs -1

Now that we have preprocessed all of our data we, can look to see how our machine learning models perform on this newly collected data by training only on the old 2014-2018 dataset. Before training a machine learning model, we will need to compute a way of ranking the features a prior. To do so we can execute ``genolearn.feature-selection`` which requires ``data_config`` and ``model_config`` files. Here is an example of generating the contents for both:

.. code-block:: bash

    python3 -m genolearn.utils.make_config model_config.json '{"n_jobs" : -1, "class_weight" : "balanced", "max_depth" : range(5, 105, 5), "random_state" : range(10)}'
    python3 -m genolearn.utils.make_config data_config.json '{"path" : "data", "meta_path" : "raw-data/meta-data.csv", "identifier" : "Accession", "target" : "Region", "group" : "Year"}'

The above should generate two files in your current directory

.. code-block:: text
    :caption: data_config.json

    {
        "path": data",
        "meta_path": "raw-data/meta-data.csv",
        "identifier": "Accession",
        "target": "Region",
        "group": "Year"
    }

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

.. code-block:: bash

    python3 -m genolearn.train output RandomForestClassifier data_config.json model_config.json -train 2018 -test 2019 -K 10000 -order fisher-scores.npz -order_key 2018

The above generates the following tree

.. code-block:: text
    
    output
    ├── log.txt
    ├── model.pickle
    ├── params.json
    ├── predictions.csv
    └── results.npz

where 

+ log.txt contains information about running ``genolearn.train``
+ model.pickle is a saved model which achieved the highest metric and can be loaded with the ``load`` function in ``genolearn.model``
+ params.json is a json file storing the model parameters used when instantiating the model class
+ predictions.csv is a csv file with an index column containing identifiers and a single column containing model predictions from the previous model.pickle file
+ results.npz contains predictions from all model configurations and can be opened within Python as shown below

.. code-block:: python

    import numpy as np

    npz = np.load('output/results.npz')

    identifiers = npz['identifiers'] # shape (n,)
    predictions = npz['predict']     # shape (1, 3, 10, n) (1 K, 3 max_depth, and 10 random_state parameters for n predictions)
    times       = npz['times']       # shape (1, 3, 10, 2) (last dimension measures training and predicting times)

where :math:`n` is the number of valid testing strains present in 2019 i.e. only the strains in 2019 where the associated target region was present in 2018. This ignores strains in 2019 where the associated target region does not appear in 2018 as there are no training examples to learn from.




