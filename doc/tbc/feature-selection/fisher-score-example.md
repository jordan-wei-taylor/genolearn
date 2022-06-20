.. _FisherScoreExample:

Fisher Score Example
====================

In this example we compute the Fisher Scores as the number of training points increase. This example scenario involves imagining we have an increasing amount of data as we include data from previous years. Since our dataset consists of the strains between the years 2014 - 2019, we imagine ourselves at the end of 2018 and want to compute predictions for 2019 data. We further imagine we had only starting collecting data from 2018, perform the Fisher Score computations, then imagine ourselves having collected the previous year's data and repeat. That is, we compute the Fisher Scores on the following years worth of data:

+ 2018 - 2018
+ 2017 - 2018
+ 2016 - 2018
+ 2015 - 2018
+ 2014 - 2018

i.e. we will have 6 different sets of Fisher Scores where the first is only based on 2018 data, and the last based on 2014 - 2018 data.

Recall the Fisher Score for the :math:`i`-th feature can be computed with the following formula

.. math::

    S_i = \frac{\sum_j n_j(\mu_{ij} - \mu_i)^2}{\sum_j n_j\sigma_{ij}^2}

Instead of computing each of the RHS terms directly, we can instead incrementally compute the count, the sum, and the sum of squares. To compute the mean, we would have to divide the sum by the associated count, and to compute the variance we can use :math:`\sigma_{ij}^2 = \frac{1}{n}\sum_k x_k^2 - \big[\frac{1}{n}\sum_k x_k\big]^2 = \mathbb{E}[X^2] - \mathbb{E}[X]^2` where :math:`x_k` denotes the :math:`k`-th observation.

The ``inner_loop`` function below performs the incremental updates as previously described. The ``outer_loop`` function uses the statistics the ``inner_loop`` computes to compute the Fisher Score thus far.

<<<<<<< HEAD
.. literalinclude:: ../../../src/genolearn/feature_selection/fisher.py
    :caption: genolearn.feature_selection.fisher.py
    :linenos:

The above code is run by 

.. literalinclude:: ../../../src/genolearn/feature_selection/__init__.py
    :linenos:
=======
.. literalinclude:: 
    ../../../src/genolearn/feature_selection/fisher.py
    


The above code is run by 

.. literalinclude:: 
    ../../../src/genolearn/feature_selection/__init__.py
    
>>>>>>> 5f3a4fd (rebased)
