Metrics
##########################

.. warning::

    Further metrics are being developed.

Intra-class Accuracies
======================

For classification tasks, when the target labels are balanced, it is common to report accuracy as a measure of model performance. That is, the number of correct predictions divided by the total number of predictions. When the target labels are unbalanced, this measure is skewed. Consider an example where 90% of the data belongs to label 1 and 10% of the data belongs to label 2, if we had a model that predicts everything as label 1, the model would have an accuracy of 90%.

We define the intra-class accuracy to be

.. math::

    \text{acc}(\mathbf{y}, \mathbf{\hat{y}}, j) = \frac{\sum_{i : y_i = j} 1_{\hat{y}_i = j}}{\sum_{i : y_i = 1} 1},

where

.. math::

    1_{\hat{y}_i = j} = \begin{cases}1, &\text{if } i = j,\\0, & \text{otherwise}.\end{cases}