.. _MachineLearning:

Machine Learning
##########################

.. code-block:: python

    from genolearn.models.classification import RandomForestClassifier
    from genolearn.metrics               import confusion_matrix
    from genolearn.models                import save_model, feature_importance
    from genolearn                       import DataLoader
    
    dataloader = DataLoader('data', meta_path = 'meta.csv', identifier = 'identifier', group = 'year', target = 'target')

    # train on 2014-2020 and test on 2021
    X_train, Y_train = dataloader.load(*range(2014, 2021))
    X_test , Y_test  = dataloader.load(2021)

    # create an instance of a model with particular parameters from the above class
    model = RandomForestClassifier(n_estimators = 100, max_depth = 25)

    # assuming we have already defined the training data (X_train, Y_train) and testing data (X_test, Y_test)
    # fit the model to the training data
    model.fit(X_train, Y_train)

    # compute and store predictions on observations
    predictions = model.predict(X_test)

    # save model
    save_model(model, 'random-forest-v1')
    
    # show a confusion matrix
    confusion_matrix(Y_test, predictions)

    # get the feature importances the model has learnt
    feature_importance(model)

