Fisher Scores for Feature Selection
###################################

GenoLearn offers the Fisher Score feature selection method which computes a score for each feature and selects the :math:`k` highest scoring features. The Fisher Score, as taken from Aggarwal 2014 [#fisher]_, is computed by the following equation

.. math::

    S_i = \frac{\sum_j n_j(\mu_{ij} - \mu_i)^2}{\sum_j n_j\sigma_{ij}^2}

where

+ :math:`n_j` is the number of observations belonging to the :math:`j`-th class
+ :math:`\mu_j` is the global mean of the :math:`i`-th feature
+ :math:`\mu_{ij}` is the mean of the :math:`i`-th feature belonging to the :math:`j`-th class
+ :math:`\sigma_{ij}^2` is the variance of the :math:`i`-th feature belonging to the :math:`j`-th class

The above can be vectorized by the following operation :math:`\mathbf{D}\mathbf{n}\ /\ \Sigma\mathbf{n}` where :math:`ij`-th element of :math:`\mathbf{D}` is :math:`(\mu_{ij} - \mu_i)^2` and the :math:`ij`-th element of :math:`\Sigma` is :math:`\sigma_{ij}^2`. Intuitively, the Fisher Score yields a higher score if the local mean is more different to the global mean scaled by the local variation.

Example
-------

.. math::

    \begin{align*}
    & \quad \overbrace
    {\begin{bmatrix}
        0 & 0 & 1 & 0 & 0\\
        1 & 1 & 1 & 0 & 1\\
        0 & 1 & 0 & 1 & 0\\
        0 & 3 & 0 & 1 & 1\\
        1 & 3 & 1 & 0 & 1\\
        0 & 1 & 1 & 1 & 0\\
        1 & 0 & 0 & 1 & 3\\
        2 & 2 & 4 & 2 & 0\\
        0 & 0 & 1 & 1 & 0\\
        0 & 1 & 0 & 1 & 2
    \end{bmatrix}}^\mathbf{X}
    \hspace{14em}
    \overbrace
    {\begin{bmatrix}
        0\\
        1\\
        0\\
        1\\
        1\\
        2\\
        0\\
        2\\
        0\\
        0
    \end{bmatrix}}^\mathbf{y}\\\\
    \mathbf{n} &= \begin{bmatrix}5 & 3 & 2\end{bmatrix} && \begin{bmatrix}\text{count of } 0\text{s} & \text{count of } 1\text{s} & \text{count of } 2\text{s}\end{bmatrix}\\\\
    \mathbf{D} &= \begin{bmatrix}
                        0.09      & 0.64      & 0.25      & 0.        & 0.04     \\
                        0.0277778 & 1.2844444 & 0.0544444 & 0.2177778 & 0.04     \\
                        0.25      & 0.09      & 2.56      & 0.49      & 0.64
                  \end{bmatrix} && \begin{matrix}(\text{mean when } \mathbf{y} = 0 \text{ minus global mean squared})\\
                                                (\text{mean when } \mathbf{y} = 1 \text{ minus global mean squared})\\
                                                (\text{mean when } \mathbf{y} = 2 \text{ minus global mean squared})\end{matrix}\\\\
    \Sigma &= \begin{bmatrix}
                0.16      & 0.24      & 0.24      & 0.16      & 1.6      \\
                0.2222222 & 0.8888889 & 0.2222222 & 0.2222222 & 0.       \\
                1.        & 0.25      & 2.25      & 0.25      & 0.
              \end{bmatrix}  && \begin{matrix}(\text{variance when } \mathbf{y} = 0)\\
                                             (\text{variance when } \mathbf{y} = 1)\\
                                             (\text{variance when } \mathbf{y} = 2)\end{matrix}
    \end{align*}

resulting in 

.. math::

    \mathbf{S} = \begin{bmatrix} 0.2980769 & 1.6564885 & 1.026178 & 0.8305085 & 0.2\end{bmatrix}


For this example, the feature rankings are :math:`[2, 3, 4, 1, 5]` i.e. the second feature is the most important and the fifth feature is the least important when ranked according to their associated Fisher Scores.

.. [#fisher] Charu C. Aggarwal. 2014. Data Classification: Algorithms and Applications (1st. ed.). Chapman & Hall/CRC. page 44
