GenoLearn Feature Importance
############################

After executing the ``train`` command, it may be of interest to examine which genome sequences are important as determined by the trained Machine Learning model. Upon executing the ``feature-importance`` command, the user is prompted to select a subdirectory in the ``train`` directory within their ``working directory``. A new subdirectory ``importance`` is generated containing four files. The ``csv`` file contains the genome sequences ranked from most important to least important. The ``npz`` file contains more information including the actual feature importance scores for each genome sequence. File names starting with ``full-`` correspond to the model that was trained on both the *train* and *validation* datasets instead of only the *train* dataset. To access the ``npz`` file the user must run in Python

.. code-block:: python

    import numpy as np

    # object is similar to a dictionary
    npz = np.load('path/to/file.npz', allow_pickle = True)

    # prints the keys ['features', 'ranks', 'aggregate_scores', 'detailed_scores']
    print(list(npz))

    # access the aggregate scores array
    print(npz['aggregate_scores'])

The scores are computed by model basis and so the format of the ``detailed_scores`` array may be very different. For example, the Logistic Regression model would have a 2D array with shape ``(<no. of class labels>, <no. of features>)`` corresponding to how a feature contributes to the log-odds of belonging to a class label.