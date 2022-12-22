GenoLearn Feature Importance
############################

After executing the ``train`` command, it may be of interest to examine which genome sequences are important as determined by the trained Machine Learning model. Upon executing the ``feature-importance`` command, the user is prompted to select a subdirectory in the ``train`` directory within their ``working directory``. A new subdirectory ``importance`` is generated containing two files. The ``csv`` file contains the genome sequences ranked from most important to least important. The ``npz`` file contains more information including the actual feature importance scores for each genome sequence. To access the ``npz`` file the user must run in Python

.. code-block:: python

    import numpy as np

    # object is similar to a dictionary
    npz = np.load('path/to/file.npz')

    # print the keys ('features', 'ranks', 'scores')
    print(list(npz))

    # access the scores array
    print(npz['scores'])