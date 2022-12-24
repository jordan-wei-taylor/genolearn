GenoLearn Model Config
######################

Users need to generate a ``model config`` before they can ``train`` their Machine Learning models. A ``model config`` describes the parameters for a Machine Learning model they would like to use. By default, all settings have been set to ``scikit-learn`` default values. If it is of interest to try out a few different values for the same parameter, the user can enter when prompted

.. code-block:: text

    param prompt [default value] : value1, value2, value3, ....

if they have specific values in mind or if their values are equidistant the user can make use of the built in ``range`` function

.. code-block:: text

    param prompt [default value] : range(start, end, step)

Upon executing the ``model-config`` command option, the user is prompted to configure one of two Machine Learning models

.. code-block:: text

    Genolearn ({VERSION}) Command Line Interface

    GenoLearn is designed to enable researchers to perform Machine Learning on their genome
    sequence data such as fsm-lite or unitig files.

    See https://genolearn.readthedocs.io for documentation.

    Working directory: {WORKING_DIRECTORY}

    Command: model-config

    Select a model to configure 

    0.  back                                goes to the previous command

    1.  logistic_regression                 
    2.  random_forest                       

See the ``scikit-learn`` documentation for `Logistic Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ or `Random Forest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_ for more details.

Upon a successful execution, the an example directory tree will look like

.. code-block:: text
    :emphasize-lines: 10, 11

    working directory
    ├── data
    │   ├── genome-sequence-data.txt.gz
    │   └── metadata.csv
    ├── feature-selection
    │   ├── default-fisher
    │   └── default-fisher.log
    ├── meta
    │   └── default
    ├── model
    │   └── random-forest
    └── preprocess
       ├── dense   [.npz files]
       ├── features.txt.gz
       ├── info.json
       ├── meta.json
       ├── preprocess.log
       └── sparse  [.npz files]
