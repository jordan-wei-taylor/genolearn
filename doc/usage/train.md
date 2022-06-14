Train
#####

Creates a numpy ``npz`` object within a preprocessed directory. The npz object is similar to that of a dictionary where the ``key`` contains the groups of the observations and the ``value`` is a 1-dimensional array with the number of elements to be the number of genome sequences present in the genomoe sequence data used to preprocess.

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