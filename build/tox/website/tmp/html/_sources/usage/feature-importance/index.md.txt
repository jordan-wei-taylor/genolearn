.. _FeatureImportance:

Feature Importance
##########################

..
    .. code-block:: python

        # import function to load model from path and function that returns the feature importance
        from biolearn.models import load_model
        from biolearn import feature_importance

        # load model from previous example
        model = load_model('random-forest')

        # prints the feature importance for the trained random forest
        feature_importance(model)

