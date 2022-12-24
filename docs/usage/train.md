GenoLearn Train
###############

Fits Machine Learning models to their genome sequence data. Users are initially prompted to select a ``feature selection`` file to use, which has an associated ``metadata`` file, and a ``model config`` file before being prompted for further information. An example prompt is shown below.

.. rubric:: Prompted Information
.. code-block:: text

    Genolearn ({VERSION}) Command Line Interface

    GenoLearn is designed to enable researchers to perform Machine Learning on their genome
    sequence data such as fsm-lite or unitig files.

    See https://genolearn.readthedocs.io for documentation.

    Working directory: {WORKING_DIRECTORY}

    Command: train

    Train parameters for metadata "default" with feature-selection "default-fisher" and model config "random-forest"

    output_dir [default-fisher-random-forest]           : 
    num_features* [1000]                                : 
    binary [False]                                      :
    min_count [0]                                       : 
    target_subset [None]                                : 
    metric [f1_score]                                   : 
    aggregate_func {mean, weighted_mean} [weighted_mean]: 

where

+ ``output_dir`` is the subdirectory name to output contents to within ``<working directory>/train``. By default this is ``<feature-selection>-<model-config>``.
+ ``num_features*`` is a sequence of comma seperated integers indicating how many sequence features to use when training. For example, if ``num_features`` is set to ``100, 1000``, GenoLearn will perform a grid parameter search when using the 100 highest scoring genome sequences (according) to the Fisher Score and then again using 1000 genome sequences.
+ ``binary`` is a flag for converting the count data to binary data. This results in sequence *presence* modelling instead of sequence *count*.
+ ``min_count`` states the number of times a class label has to appear in the *train* set of the metadata for instances of that class label to be modelled. For example, setting ``min_count`` to 10 results in instances being ignored of the associated target label appears less than 10 times.
+ ``target_subset`` is a comma seperated string of target class labels to model. For example. if in your metadata, your class labels are a, b, c, d, and e, and we only want to model a, b, c then we would set ``target_subset`` as a, b, c
+ ``metric`` is the objective function to evaluate goodness-of-fit to select the best performing model in your parameter gridsearch. See `Metrics <../background/metrics.html>`_ for more details.
+ ``aggregate_func`` is set to either mean or weighted\_mean to best aggregate the previously set ``metric`` for each class label.

A subdirectory within the ``train`` directory is generated with contents

.. code-block:: text

   <output directory>
    ├── encoding.json      # translation between the prediction integers and the class labels
    ├── full-model.pickle  # trained scikit-learn model on the full train / val datasets with best parameters found in the gridsearch
    ├── model.pickle       # trained scikit-learn model which performed the best during the gridsearch
    ├── params.json        # parameters of the saved model
    ├── predictions.csv    # predictions on the validation dataset with probabilities if applicable
    ├── results.npz        # numpy file with more information on the training such as predictions for all models in the gridsearch, train / validation time etc
    └── train.log          # text file logging information such as time of execution, RAM usage, parameters for command execution

