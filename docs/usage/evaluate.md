GenoLearn Evaluate
##################

Users may wish to ``evaluate`` their trained model on a different part of their dataset. Suppose the user would like to ``evaluate`` their model on data points that have not been labelled in the ``metadata`` csv. Upon executing the ``evaluate`` command, the user is prompted to select a ``train`` subdirectory and then is prompted for

.. code-block:: text

    output filename :
    group values*   :

where

+ ``output filename`` is, as the name suggests, the output filename such that ``<output filename>.csv`` and ``<output filename>.npz`` will be generated within a subdirectory ``evaluate`` within their chosen ``train`` subdirectory.
+ ``group values*`` is a comma seperated string of group values (valid group values will be shown). Only data points corresponding to the ``group values*`` will be evaluated in the output files.

Upon a successful execution of the ``evaluate`` command, the following should be generated within the user selected ``train`` subdirectory

.. code-block:: text

    evaluate
    ├── <output>.csv      
    └── full-<output>.csv 

where both files contain predictions on instances associated with the entered ``group values*``. The first csv corresponds to the model in the parent directory trained only on the *train* dataset whilst the csv starting with ``full-`` corresponds to the model in the parent directory trained on both the *train* and *validation* datasets.