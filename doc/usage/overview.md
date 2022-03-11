Overview
########

Data
====

To use GenoLearn once you have installed it, we require you to first preprocess your genome sequence files (such as an fsm-lite gun-zipped) into a more readable format. This requires executing the package on a .gz file specifying any options. The .gz file should be a text file with the following format:

.. code-block:: text

    feature_1 | observation_{1}:count_{1,1} observation_{1}:count_{2,1} ...
    feature_2 | observation_{1}:count_{1,2} observation_{1}:count_{3,2} ...
    ...

Preprocessing
=============

First you will need to preprocess your data into a format this package has been optimized for. Depending on the magnitude of your data and the specifications of your machine, this may take minutes or even hours.

The preprocessing writes to a directory specified by the user. By default, the directory will be

.. code-block:: text
    
    DATASET NAME
    ├── dense
    |   ├── *.npz
    ├── feature-selection
    ├── sparse
    |   ├── *.npz
    ├── features.txt
    ├── log.txt
    └── meta.json

+ `dense` folder contains dense arrays
+ `sparse` folder contains sparse arrays
+ `feature-selection` folder is initially empty but can be populated using the later discussed `feature_selection.py` module
+ `features.txt` contains all the genome sequences seperated by a single empty space
+ `log.txt` contains the parameters used to generate the current folder, the timestamp of the execution, and the RAM usage
+ `meta.json` contains the number of samples $n$, the number of genome sequences $m$ and the maximum value observed *max*

The `dense` folder contains dense arrays and the `sparse` folder contains the same information but as sparse arrays. The `feature-selection` folder is initially empty and can be populated using the later discussed `feature_selection` module. `features.txt`  

On our **\<NAME OF DATASET WE USE>** dataset, this meant reducing data reading time from hours, in the case of the raw fsm-lite file, to minutes,  in the case of our preprocessed directory.

See :ref:`Preprocessing <Preprocessing>` for more details.

Data Loader
==========

A ``DataLoader`` class can be found in ``genolearn.dataloader``. This expects a path to the previous step's preprocessed directory. The DataLoader class supports returning a dense or sparse matrix for the observations. Later Machine Learning models supports both sparse and dense matrices.

See :ref:`Data Loader <DataLoader>` for more details.

Feature Selection
=================

As the number of genome sequences tends to be large, we need to perform feature selection to then constrain our later Machine Learning models' complexity. This optional step is to remove features with high measures of similarity. 

See :ref:`Feature Selection <FeatureSelection>` for more details.


Machine Learning
================

Within ``biolearn.models``, there are machine learning models from the popular library ``sklearn`` and a few useful functions to save / load models.

Feature Importance
==================

After fitting a model, the user may wish to identify which were the most importance features in their dataset. With ``biolearn.feature_importance``, the user can simply call this function on a fitted model and analyse the feature importance based on the particlar model. For example, Logistic Regression will return a set of coefficients for each feature corresponding to the contribution to each class label.

