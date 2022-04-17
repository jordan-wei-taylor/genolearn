.. _FeatureSelection:

Feature Selection
##########################

For genome count datasets, it is often the case that the number of sequences is large. This presents memory problems when trying to read the data. Assuming the dataset originally has :math:`n` observations each having :math:`m` genome sequence counts, the total memory cost would be of the order :math:`n\times m`. Feature selection is a process that aim to pre-select :math:`k` features from :math:`m` before conducting any further analysis and experimentation where :math:`k` is much smaller than :math:`m`.


Fisher Score
============

``genolearn`` offers the Fisher Score feature selection method which computes a score for each feature and selects the :math:`k` highest scoring features. The Fisher Score, as taken from REFERENCE, is computed by the following equation

.. math::

    S_i = \frac{\sum_j n_j(\mu_{ij} - \mu_i)^2}{\sum_j n_j\sigma_{ij}^2}

where

+ :math:`n_j` is the number of observations belonging to the :math:`j`-th class
+ :math:`\mu_j` is the global mean of the :math:`i`-th feature
+ :math:`\mu_{ij}` is the mean of the :math:`i`-th feature belonging to the :math:`j`-th class
+ :math:`\sigma_{ij}^2` is the variance of the :math:`i`-th feature belonging to the :math:`j`-th class

The above can be vectorized by the following operation :math:`\mathbf{n}\mathbf{D}\ /\ \mathbf{n}\Sigma` where :math:`ij`-th element of :math:`\mathbf{D}` is :math:`(\mu_{ij} - \mu_i)^2` and the :math:`ij`-th element of :math:`\Sigma` is :math:`\sigma_{ij}^2`. Intuitively, the Fisher Score yields a higher score if the local mean is more different to the global mean scaled by the local variation.

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


Custom Feature Selection
------------------------

``genolearn`` offers the user the ability to create their own custom feature selection process. One needs to define ``init``, ``inner_loop``, and ``outer_loop`` functions in their custom module. The ``base_feature_selection`` method is then executed with those modules' functions. Refer to the example to see how the Fisher Score is computed as a guideline when creating your own.

.. literalinclude:: ../../../genolearn/feature_selection/__init__.py
    :linenos:
    
Feature Selection Execution
---------------------------

.. code-block:: text

    python -m genolearn.feature_selection --help
    usage: __main__.py [-h] [-group GROUP] [-method METHOD] [--sparse] output path meta_path identifier target values [values ...]

    Generates an ordered list of features and meta information.

    Example
    =======

    >>> # fisher score default
    >>> python -m genolearn.feature_selection fisher-scores.npz data raw-data/meta-data.csv Accession Regions 2014 2015 2016 2017 2018 2019 -group Year

    >>> # custom (expected custom.py)
    >>> python -m genolearn.feature_selection custom-scores.npz data raw-data/meta-data.csv Accession Regions 2014 2015 2016 2017 2018 2019 -group Year -method custom

    positional arguments:
    output          output file name
    path            path to preprocessed directory
    meta_path       path to meta file
    identifier      column of meta data denoting the identifier
    target          column of meta data denoting the target
    values          incremental identifiers (or groups) to perform feature selection on

    optional arguments:
    -h, --help      show this help message and exit
    -group GROUP    column of meta data denoting the grouping of labels
    -method METHOD  either "fisher" for built in Fisher Score or a module name (see example)
    --sparse        if sparse loading of data is preferred

.. toctree::
    :hidden:

    example