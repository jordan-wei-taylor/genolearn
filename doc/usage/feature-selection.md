Feature Selection
#################

Creates a numpy ``npz`` object within a preprocessed directory. The npz object is similar to that of a dictionary where the ``key`` contains the groups of the observations and the ``value`` is a 1-dimensional array with the number of elements to be the number of genome sequences present in the genomoe sequence data used to preprocess.

Suppose your meta-data file had the form

.. list-table:: Example meta-data.csv
   :header-rows: 1
   :align: center

   * - Accession
     - Year
     - Region
   * - SRR001
     - 2014
     - UK
   * - SRR002
     - 2014
     - Middle East
   * - ...
     - ...
     - ...
   * - SRR999
     - 2019
     - C\. Europe


Example feature selection usage for the above dataset if we wanted to create a score for all data points belonging to the year 2018 only

.. code-block:: python

    output='fisher-scores.npz'
    data='data'
    meta='meta-data.csv'
    identifier='Accession'
    target='Regions'
    values='2018'
    group='Year'

    python3 genolearn.feature_selection $output $data $meta $identifier $target $values -group $group


This would create the file ``data/feature-selection/scores.npz`` using the preprocessed data found in ``data/dense`` and the meta data ``meta-data.csv`` where the identifying column is ``Accession``, the target column ``Region``, the grouping column of the identifiers ``Year``, and the identifier (or group) ``2018``. The resulting ``scores.npz`` would contain a single key ``2018`` with the associated 1D-array containing a score for each genome sequence. By default, the score values are the Fisher Scores but the user can implement their own scoring regime following the example.

To extend the example, if we would like to create a Fisher Score for each genome sequence for the periods 2018-2018, 2017-2018, ...,2014-2018 we would execute


.. code-block:: bash

    output='fisher-scores.npz'
    data='data'
    meta='meta-data.csv'
    identifier='Accession'
    target='Regions'
    values='2018 2017 2016 2015 2014'
    group='Year'
    python3 -m genolearn.feature_selection $output $data $meta $identifier $target $values -group $group

Executing the above would create the file ``data/feature-selection/fisher-scores.npz`` based on a meta data file ``meta-data.csv`` which has columns "Accession", "Regions" and "Year". It computes the Fisher Score for data that belongs to the years 2018-2018, 2017-2018, ...,2014-2018.

The input parameters for ``genolearn.feature_selection`` are

.. list-table:: Parameters for genolearn.feature_selection
   :widths: 25 25 50 100
   :header-rows: 1
   :align: center

   * - Parameter
     - Flag
     - Default
     - Description
   * - output
     - 
     - 
     - output file name
   * - path
     - 
     - 
     - path to preprocessed directory
   * - meta_path
     -
     - 
     - path to meta data csv
   * - identifier
     - 
     - 
     - meta data column denoting the identifier
   * - target
     - 
     - 
     - meta data column denoting the target
   * - values
     - 
     - 
     - incremental identifiers (or groups) to perform feature selection on
   * - group
     - \-group
     - None
     - meta data column denoting the groupings of the observations
   * - method
     - \-method
     - fisher
     - either "fisher" for built-in Fisher Score or a module name (see example in documentation)
   * - log
     - \-log
     - None
     - name of the log file generated upon completion
   * - sparse
     - \-\-sparse
     - False
     - indicate if sparse loading of the data is preferred


Why we were interested in analysing this growing period is to investigate the effect of having more data when performing our supervised learning task. So the scenarios we were modelling were:
+ the current year is 2018 and we only have the current year's worth of data
+ the current year is 2018 and we have the past two year's worth of data
+ ...

We then made predictions on the 2019 data to see how measures such as recall changed.
For genome sequence data, it is often the case that the feature space is large and therefore, loading it all into memory for further analysus is not feasible. Instead, we can compute a measure of a priori relatedness to our target variables of interest and only use the top :math:`k` features based on this measure of relatedness. ``GenoLearn`` provides, by default, a feature selection framework which uses the Fisher Score as the measure of relatedness by default. See :ref:`FisherScoreExample` for how it is defined and further details.


Custom Feature Selection
========================

If the user would like to define their own way of computing relatedness, then it is recommended to see how the Fisher Score is defined within ``GenoLearn`` and create their own following the template used.

.. literalinclude:: ../../src/genolearn/feature_selection/fisher.py
