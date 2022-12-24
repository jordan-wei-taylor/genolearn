GenoLearn Command Overview
##########################

Users initially will only be able to execute the Setup command. GenoLearn makes available new commands as the user executes the currently available commands.


genolearn-setup
===============

Prepares the ``current directory`` and the ``working directory`` to be used by GenoLearn. Ensure that you have your ``genome sequence data`` (\*.gz) and ``metadata`` (\*.csv) in a ``data directory``.

.. code-block:: text
    
    <current directory>
    └── <data directory>
        ├── <genome sequence data>
        └── <metadata>

During the setup, you will be prompted to first select which subdirectory within your ``current directory`` is your ``data directory`` and then which file within your ``data directory`` is your ``metadata``. 

genolearn-clean
===============

Deletes all GenoLearn generated directories and files in the ``working directory``. Users may wish to start their project again. This process cannot be undone and the user will be asked to confirm first before the command is executed.


genolearn
=========

The main command which enables you to access the later subcommands. Users should execute this command when in a ``working directory``. Should the user execute ``genolearn`` not in a ``working directory``, it will assume the last ``working directory`` it was executed in.

print
-----

Prints various GenoLearn generated files. Initially the user can only choose from history and config which corresponds to the history of GenoLearn commands executed in the ``working directory`` and the configuration used by GenoLearn defining the data directory and the metadata file within it. As the user executes later commands, they will be able to additionally select to print or analyse their preprocessed metadata and to print

- preprocessed metadata stored in the ``meta`` subdirectory
- model configurations stored in the ``model`` subdirectory
- generated files that end in ``.log``

preprocess
----------

Preprocess subcommand of genolearn. This subcommand preprocesses the user's data into a more friendly format which is faster to read. 


.. rubric:: preprocess sequence

Preprocesses sequence data and generates the ``preprocess`` subdirectory. The user has to first select which ``*.gz`` file within the ``data directory`` to preprocess. If there is only a single ``*.gz`` file within the ``data directory``, it will be automatically selected. Upon selecting a sequence file, the user is prompted for more information:

.. code-block:: text

    batch-size   : number of concurrent observations to preprocess at the same time
    n-processes  : number of parallel processes
    sparse       : flag to generate sparse data representation
    dense        : flag to generate dense  data representation
    verbose      : integer denoting the number of features between each verbose update
    max_features : integer denoting the first number of features to preprocess (leave as None to preprocess all)


.. rubric:: preprocess combine


Similar to ``preprocess sequence``, preprocesses another ``*.gz`` file and combines the outputs with the previous execution of ``preprocess sequence``. 

.. code-block:: text

    batch-size   : number of concurrent observations to preprocess at the same time
    n-processes  : number of parallel processes
    verbose      : integer denoting the number of features between each verbose update


.. note::

    This command is only available if the user has previously executed ``preprocess sequence`` in the ``working directory``.


.. rubric:: preprocess meta


Preprocesses the ``metadata`` defining the ``identifier`` column, ``target`` column, and splitting which datapoints can be used to train and validate our later Machine Learning models on. The user is prompted on

.. code-block:: text

    output              : filename of preprocessed metadata
    identifier          : identifier column in input metadata
    target              : target column in input metadata
    group               : group column in input metadata
    train group values* : group values to assign as train data       [if group  = None]
    val group values *  : group values to assign as validation data  [if group  = None]
    proportion train    : proportion of data to assign as train      [if group != None]

The user should leave the ``group`` value as ``None`` if they want to randomly assign their data as train / validation (this is standard practice in the Data Science community). If the user has defined their own groupings of the data points, they should specify the group column and state which (non-overlapping) group values belong to the train and validation datasets. 

.. note::

    This command is only available if the user has previously executed ``preprocess sequence`` in the ``working directory``.


feature-selection
-----------------

Compute a score for each feature quantifying some measure to later select which features to use for modelling purposes. By default, the Fisher Score for Feature Selection as described by Aggarwal 2014 [#fisher]_ is used. The user is prompted for

.. code-block:: text

    name      : output file name

The user can select their own ``custom`` method so long as they have a \<custom\>.py file with certain functions defined located in the ``current directory`` or the ``working directory``. See here for more details.

.. note::

    This command is only available if the user has previously executed ``preprocess meta`` in the ``working directory``.


model-config
------------

Creates a Machine Learning model configuration file used by the later ``train`` command. The user is prompted on which classifier they would like to design a config file for before prompted for further information on each of the chosen model's hyperparameters. The user can choose to perform a gridsearch by entering multiple values seperated by a comma. 

.. note::

    This command is only available if the user has previously executed ``feature-selection`` in the ``working directory``.


train
-----

Trains a Machine Learning model based on previous generated configuration files. Model is tuned based on the hyperparameter gridsearch defined in the ``model-config`` execution and generates the following subdirectory within ``<working directory>``/train

.. code-block:: text

   <output directory>
    ├── encoding.json      # translation between the prediction integers and the class labels
    ├── full-model.pickle  # trained scikit-learn model on the full train / val datasets with best parameters found in the gridsearch
    ├── model.pickle       # trained scikit-learn model which performed the best during the gridsearch
    ├── params.json        # parameters of the saved model
    ├── predictions.csv    # predictions on the validation dataset with probabilities if applicable
    ├── results.npz        # numpy file with more information on the training such as predictions for all models in the gridsearch, train / validation time etc
    └── train.log          # text file logging information such as time of execution, RAM usage, parameters for command execution

The user is prompted for

.. code-block:: text

    output directory      : output directory
    num features*         : comma seperated integers for varying number of features to consider
    min count             : min count of training class observations to be considered
    target subset         : subset of target values to train on if provided
    metric                : statistical metric to measure goodness of fit
    aggregation function  : aggregation function to compute overall goodness of fit


.. note::   
    
    This command is only available if the user has previously executed ``model-config`` in the ``working directory``.


feature-importance
------------------

Given a subdirectory generated by the ``train`` command, generates an ``importance`` subdirectory describing the important features as inferred by the trained model.

.. note::   
    
    This command is only available if the user has previously executed ``train`` in the ``working directory``.

Evaluate
--------

Given a subdiirectory generated by the ``train`` command, evaluates the saved model on a subset of the user's preprocessed data.

.. note::   
    
    This command is only available if the user has previously executed ``train`` in the ``working directory``.


.. [#fisher] Charu C. Aggarwal. 2014. Data Classification: Algorithms and Applications (1st. ed.). Chapman & Hall/CRC. page 44