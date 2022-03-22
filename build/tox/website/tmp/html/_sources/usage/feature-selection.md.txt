.. _FeatureSelection:

Feature Selection
##########################

As the number of genome sequences is typically very large, it is a common challenge to load the data in its entirety, a bigger problem to perform analysis on. As a means of reducing the space of genome sequences to consider we offer the user a means of genome sequence selection prior to any further analysis. 

We offer the user, by default, a means of performing feature selection by using the Fisher Score for each genome sequence. It is defined by

.. math::
    :nowrap:
    :label: fisher

    \begin{equation}
        S_j = \frac{\sum_{c = 1}^C n_c (\mu_{jc} - \mu_j)^2}{\sum_{c = 1}^C n_c \sigma_{jc}^2}
    \end{equation}

where:

  + :math:`S_j` is the score for the :math:`j`-th genome sequence
  + :math:`C` is the number of classes (labels)
  + :math:`n_c` is the number of instances in the :math:`c`-th class
  + :math:`\mu_{jc}` is the mean of the :math:`j`-th sequence values within the :math:`c`-th class
  + :math:`\mu_{j}` is the mean of the :math:`j`-th sequence values
  + :math:`\sigma_{jc}^2` is the variance of the :math:`j`-th sequence values within the :math:`c`-th class

The intuition behind the use of Eq. :eq:`fisher` is to quantify the the ratio between the weighted differences in intra class mean values to the global mean :math:`n_c(\mu_{jc} - \mu_j)^2` and the weighted variances :math:`n_c \sigma_{jc}^2`. That is, if a feature has very different locations for each class, it would have a high Fisher Score. In contrast, if the feature had similar values irrespective of class label, it would have a lower Fisher Score.