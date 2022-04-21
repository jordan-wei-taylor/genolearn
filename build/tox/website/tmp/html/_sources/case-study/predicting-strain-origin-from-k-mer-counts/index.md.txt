.. _PaperA:

Predicting Strain Origin From k-mer Counts
##########################################

In this case study, we use ``genolearn`` to preprocess our :math:`k`-mer count dataset which has been processed by `fsm-lite <https://github.com/nvalimak/fsm-lite>`__. Our dataset consists of 2,474 strains, 12,103,121 unique :math:`k`-mer counts, and 12 unique origin regions. That is, our input data, :math:`\mathbf{X}`, consists of 2,474 rows and 12,103,121 columns and our target output data, :math:`\mathbf{y}`, consists of 2,474 regions which are:

+ Asia
+ Australasia
+ C America
+ C Europe
+ M East
+ N Africa
+ N America
+ N Europe
+ S America
+ S Europe
+ Subsaharan Africa
+ UK

As the feature space is large, we have employed the :ref:`Fisher Score <FisherScore>` as a means to pre-screen genome sequences that carry informative information about the strain origin regions. We consider two models; `Logistic Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__ for it's explainability, and the `Random Forest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__ for it's speed. The analysis we have conducted is to vary the number of features to use as ranked by the Fisher Score which we denote as :math:`k` and what we believe to be the main complexity parameter for each of the two previously mentioned machine learning models, ``C``, which controls the amount of regularisation for Logistic Regression, and ``max_depth`` which controls the maximum number of branches for each `Decision Tree <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`__ within the Random Forest.

.. 
  .. raw:: html

      <style>
        .img-container {
          text-align: center;
        }
      </style>
      <div class="img-container">
          <figure>
              <img src="../_static/generated/intra-accuracies.png" width=90% />
              <br><br>
              <figurecaption>Figure 1. <i>Intra-class accuracies.</i></figurecaption>
          </figure>
          <br>
      </div>


..
  .. raw:: html

      <style>
        .img-container {
          text-align: center;
        }
      </style>
      <div class="img-container">
          <figure>
              <img src="../_static/generated/times.png" width=90% />
              <br><br>
              <figurecaption>Figure 2. <i>Training and Evaluation Times.</i></figurecaption>
          </figure>
          <br>
      </div>