GenoLearn Feature Selection
###########################

For genome count datasets, it is often the case that the number of sequences is large. This presents memory problems when trying to read the data. Assuming the dataset originally has :math:`n` observations each having :math:`m` genome sequence counts, the total memory cost would be of the order :math:`n\times m`. Feature selection is a process that aim to pre-select :math:`k` features from :math:`m` before conducting any further analysis and experimentation where :math:`k` is much smaller than :math:`m`.


By default, GenoLearn provides an implementation of using Fisher Scores for Feature Selection as described in `Fisher Score for Feature Selection <../background/fisher-score.html>`_.


Custom Feature Selection
------------------------

GenoLearn offers the user the ability to create their own custom feature selection process. One needs to define ``init``, ``loop``, and ``post`` functions in their custom module. As an example, below is the Fisher Score implementation.

.. literalinclude:: ../../genolearn/core/feature_selection/fisher.py
    :linenos: