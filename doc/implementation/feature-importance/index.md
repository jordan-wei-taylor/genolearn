.. _FeatureImportance:

Feature Importance
##################

Once a machine learning model has been fitted (trained) to a dataset, it is common to examine which input features (genome sequences) the machine learning model thinks are important. The way we compute this is different per model.

Logistic Regression Example
===========================

The ``LogisticRegression`` model has a coefficient for each feature - label pair which denotes the change in the log likelihood. For example, if we had :math:`m` different genome sequences in our input data and :math:`c` different labels in our target output data, then our model would have computed a matrix of size :math:`(m,c)` where the :math:`i`-th row and the :math:`j`-th column represents the change that the :math:`i`-th genome sequence value has on the :math:`j`-th class label.


.. code-block:: python

    from genolearn.feature_importance import FeatureImportance
    
    # defined model from genolearn.models.classification trained
    model = ...

    importance = FeatureImportance(model)

    # returns a numpy array of the importance scores for each feature (different interpretation for each model)
    importance.feature_score

    # returns a numpy array of integers denoting highest to lowest feature importance for a given model
    importance.ranked_features()


