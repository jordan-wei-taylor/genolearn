GenoLearn Preprocess
####################

Users first need to ``preprocess`` both their ``sequence`` and ``meta`` data to use later commands in GenoLearn. Upon executing

.. code-block:: text

    genolearn

and selecting the ``preprocess`` option number, the user will be prompted to select a ``preprocess`` command.

.. code-block:: text

    Genolearn ({VERSION}) Command Line Interface

    GenoLearn is designed to enable researchers to perform Machine Learning on their genome
    sequence data such as fsm-lite or unitig files.

    See https://genolearn.readthedocs.io for documentation.

    Working directory: {WOKRING_DIRECTORY}

    Command: preprocess

    Select a preprocess subcommand

    1.  back                            goes to the previous command

    2.  sequence                        preprocesses sequence data
    3.  combine                         preprocesses sequence data and combines to previous preprocessing
    4.  meta                            preprocesses meta data

with ``combine`` and ``meta`` commands available upon having executed ``sequence``.

Sequence
========

The user is asked to enter the **option number** for which ``.gz`` file to preprocess within their ``data directory``. 

The user is prompted for parameters

.. code-block:: text

    batch_size [None]  : 
    n_processes [None] : 
    sparse [True]      : 
    dense [True]       : 
    verbose [250000]   : 
    max_features [None]:

where

+ ``batch_size`` determines how many concurrent identifiers to preprocess per run of the ``.gz`` file. By default, GenoLearn arbitrarily sets this to the minimum of your OS limit (RLIMIT_NOFILE) and :math:`2^14`.
+ ``n_processes`` determines how many processes to run when converting temp ``txt`` files to ``numpy`` arrays. By default, GenoLearn sets this to the number of physical CPU cores.
+ ``sparse`` flag indicates if the preprocessing should output sparse arrays.
+ ``dense`` flag indicates if the preprocessing should output dense arrays.
+ ``verbose`` determines the number of sequences to cycle through before printing a new line.
+ ``max_features`` determines the number of first number of sequences to preprocess. Mainly used for debugging purposes. Users should always leave this as the default ``None``.

Upon a successful execution, within the ``working directory`` is a ``preprocess`` subdirectory with the following tree

.. code-block:: text

    preprocess
    ├── dense   [{identifier}.npz files]
    ├── features.txt.gz
    ├── info.json
    ├── meta.json
    ├── preprocess.log
    └── sparse  [{identifier}.npz files]

.. note::

    Whilst the ``preprocess sequence`` command is being executed, it will print two numbers. The first is the number of unique identifiers found so far, and the second is the number of sequences the program has cycled over for preprocessing.

Combine
=======

This command is only available once the user has executed the ``preprocess sequence`` command. The user is asked to enter the **option number** for which ``.gz`` file to preprocess within their ``data directory``. The user cannot select the same file used during the original ``preprocess sequence``. If this is not the first time ``preprocess combine`` is being executed, the user also cannot select any file used in previous runs of ``preprocess combine``. 

The user is prompted for parameters

.. code-block:: text

    batch_size [None]  : 
    n_processes [None] : 
    verbose [250000]   : 

where the other parameters mentioned in ``preprocess sequence`` are automatically set according to the previous run of ``preprocess sequence``. Upon a successful execution, within the ``working directory`` the ``preprocess`` subdirectory now has the following tree

.. code-block:: text
    :emphasize-lines: 2, 3, 8

    preprocess
    ├── combine.log
    ├── dense   [more {identifier}.npz files]
    ├── features.txt.gz
    ├── info.json
    ├── meta.json
    ├── preprocess.log
    └── sparse  [more {identifier}.npz files]

Meta
====

This command is only available once the user has executed the ``preprocess sequence`` command. This command defines the *train* and *test* datasets for later modelling purposes.

The user is prompted for parameters

.. code-block:: text

    output        [default]:
    identifier             :
    target                 :
    group            [None]:
    proportion train [0.75]:

if the user selects ``None`` as the ``group`` value or 

    output        [default]:  
    identifier             :
    target                 :
    group            [None]:
    train group values*    :
    test  group values*    :

if the user enters a column present in their metadata csv where

+ ``output`` is the output filename storing the collected information from the user.
+ ``identifier`` is a column within the metadata csv containing the unique identifiers.
+ ``target`` is a column within the metadata csv containing the target metadata labels.
+ ``group`` is either a column within the metadata csv that helps the user split the data into *train* and *test* datasets or left as ``None``
+ ``proportion train`` is only available if ``group`` is ``None`` and is a sensible proportion value to randomly assign as *train* with the rest as *test*.
+ ``train group values*`` is only available if ``group`` is not ``None`` and contains a comma seperated string indicating which group values belong to *train*.
+ ``test  group values*`` is only available if ``group`` is not ``None`` and contains a comma seperated string indicating which group values belong to *test*. Note that these values cannot overlap with ``train group values*``.

Upon a successful execution of this command, within the ``working directory`` is a ``meta`` subdirectory with an additional entry of ``output``.
  

