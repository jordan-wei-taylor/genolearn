Evaluate
#########

Here we show how to evaluate a trained machine learning model on a specified dataset.

Suppose your trained model has the path ``path/model.pickle``, and we already have a ``data.config.json`` file then we can create ``predictions.csv`` with the following command

.. code-block:: bash

    output="predictions.csv"
    model="path/model.pickle"
    data_config="data_config.json"
    values="2019"
    feature_selection="fisher-scores.npz"
    key="2014"

    python3 -m genolearn.evaluate $output $model $data_config $values --feature-selection $feature_selection --key $key

The input parameters for ``genolearn.evaluate`` are

.. list-table:: Parameters for genolearn.evaluate
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
   * - model
     - 
     - 
     - path to model
   * - data_config
     -
     - 
     - configuration file for the ``DataLoader`` object
   * - values
     - 
     - 
     - identifiers (or group) to evaluate your specified model on
   * - feature-selection
     - --feature-selection
     - None
     - if specified, use features according to the feature-selection array
   * - K
     - --nfeatures
     - None
     - number of features to use - note it must match the expected ``model`` input i.e. the same setting of ``K`` when training - see ``params.json``.
   * - key
     - --key
     - None
     - incremental identifiers (or groups) to perform feature selection on
   * - ascending
     - --ascending
     - None
     - meta data column denoting the groupings of the observations