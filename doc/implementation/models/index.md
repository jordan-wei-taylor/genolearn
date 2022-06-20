.. _Models:

Models
##########################

``genolearn`` currently only supports popular classification models from ``sklearn``. The explicit models are

+ `Logistic Regression <https://scikit-learn.org/stable/modules/naive_bayes.html>`__
+ `Multi-Layer Perceptron <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`__
+ `k-Nearest Neighbours <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`__
+ `Support Vectors <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`__
+ `Gaussian Process <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html>`__
+ `Decision Tree <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`__
+ `Random Forest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__
+ `AdaBoost <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html>`__
+ `Gaussian Naive Bayes <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`__
+ `Linear Discriminent Analysis <https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis>`__
+ `Quadratic Discriminant Analysis <https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html>`__

all of which can be located in ``genolearn.models.classification``. For example, if you would like to use a Logistic Regression model, then the following will import statement will be required

.. code-block:: python

    from genolearn.models.classification import LogisticRegression

To list the classifiers within ``genolearn.models.classification``, we can run the following code

.. code-block:: python

    from genolearn.models import classification

    print(classification.list)
    # ['LogisticRegression', 'MLPClassifier', 'KNeighborsClassifier', 'SVC', 'GaussianProcessClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier',
    #  'AdaBoostClassifier','GaussianNB', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']
