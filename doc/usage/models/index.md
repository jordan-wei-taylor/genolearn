Machine Learning
##########################

.. code-block:: python

    # import function to get various models
    from biolearn.models  import get_model, save_model
    from biolearn.metrics import confusion_matrix

    # execute function to see list of models accepted
    print(get_model())

    # get a particular model class
    Model = get_model('random-forest')

    # create an instance of a model with particular parameters from the above class
    model = Model(n_estimators = 100, max_depth = 25)

    # assuming we have already defined the training data (X_train, Y_train) and testing data (X_test, Y_test)
    # fit the model to the training data
    model.fit(X_train, Y_train)

    # compute and store predictions on observations
    predictions = model.predict(X_test)

    # save model
    save_model(model, 'random-forest')
    
    # show a confusion matrix
    confusion_matrix(Y_test, predictions)

