.. _FeatureSelection:

Feature Selection
##########################

For genome count datasets, it is often the case that the number of sequences is large. This presents memory problems when trying to read the data. Assuming the dataset originally has :math:`n` observations each having :math:`M` genome sequence counts, the total memory cost would be of the order :math:`n\times M`. Feature selection is a process that aim to pre-select :math:`m` features from :math:`M` before conducting any further analysis and experimentation where :math:`m` is much smaller than :math:`M`.

``genolearn`` offers the Fisher Score feature selection method which gives a higher score for features that behave differently across the different associated labels.

.. math::

    S_i = \frac{\sum_j n_j(\mu_{ij} - \mu_i)^2}{\sum_j n_j\sigma_{ij}^2}

.. code-block:: text

    >>> python -m fisher --help

    usage: fisher.py [-h] input_path

        Computes the fisher score as defined in the thread <website>.

        Required Arguments
        =======================
            input_path         : path to preprocessed directory

        Example Usage
        =======================
            >>> python -m biolearn.fisher data
        

    positional arguments:
    input_path

    optional arguments:
    -h, --help         show this help message and exit
