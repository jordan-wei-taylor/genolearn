Overview
########

Data
====

To use BioLearn once you have installed it, we require you to first preprocess your fsm-lite file (gun-zipped) into a more readable format. This requires executing the package on a .gz file specifying any options. The .gz file should be a text file with the following format:

.. code-block:: text

    feature_1 | observation_{1}:count_{1,1} observation_{1}:count_{2,1} ...
    feature_2 | observation_{1}:count_{1,2} observation_{1}:count_{3,2} ...
    ...

For example suppose we have in our text file the following two lines:

.. code-block:: text

    CAAAGT | SRR1:5 SRR2:1 SRR3:2
    AAAGTA | SRR2:8

then in this example, we would have the resulting dense matrix:

.. code-block:: text

    [[5, 0],
     [1, 8],
     [2, 0]]

where the three rows correspond to the observations with the ids SRR1, SRR2, and SRR3 and the two columns representing the features "CAAAGT" and "AAAGTA", and the integer counts to be the number of times a particular feature appeared for that observation.

Preprocessing
=============

First you will need to preprocess your data into a format this package has been optimized for. Depending on the magnitude of your data and the specifications of your machine, this may take minutes or even hours.

The preprocessing writes to a directory, specified by the user, a gz file containing the dense matrix as shown above. If the groupby option has been specified, the directory will have sub-directories containing the unique values contained in the meta data with the groupby column.

This is necessary to read the data with multiple processes to leverage parallelized compute making data reading far quicker. On the DATASET NAME dataset, this meant reducing data reading time from hours, in the case of the raw fsm-lite file, to minutes,  in the case of our preprocessed directory.

DataLoader
==========

A DataLoader class can be found in ``biolearn.dataloader``. This expects a path to the previous step's preprocessed directory. The DataLoader class supports returning a dense or sparse matrix for the observations. Later models support sparse matrices and should be used to save RAM requirements at the potential cost of additional compute power.

Fisher Score
============

The fisher score is how this package reduces the feature space through feature selection. The fisher score metric was chosen as it is cheap and fast to compute. Each feature in the dataset will have an associated fisher score where the higher the score, the more *important* the feature.


Machine Learning
================

Within ``biolearn.models``, there are machine learning models from the popular library ``sklearn`` and a few useful functions to save / load models.

Feature Importance
==================

After fitting a model, the user may wish to identify which were the most importance features in their dataset. With ``biolearn.feature_importance``, the user can simply call this function on a fitted model and analyse the feature importance based on the particlar model. For example, Logistic Regression will return a set of coefficients for each feature corresponding to the contribution to each class label.

