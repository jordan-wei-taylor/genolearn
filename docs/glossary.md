Glossary
########

Supervised Learning
-------------------

Suppose we have paired labelled dataset :math:`\{(\mathbf{x}_1, \mathbf{y}_1), (\mathbf{x}_2, \mathbf{y}_2), ..., (\mathbf{x}_n, \mathbf{y}_n)\}`, where the observation :math:`mathbf{x}_i` contains :math:`m` elements and the target :math:`\mathbf{y}_i` contains :math:`c` elements. For context, :math:`m` could represent the number of unique genome sequences in our dataset, and :math:`c` could be the number of unique class labels. Supervised learning is interest in learning a function that approximates the mapping from our observation space to our target space.

Observation
-----------

An observation is the data we have access to which would be an input to a machine learning model. For example, it could be data regarding genome sequences. It is normally a row in a dataset.


Target
------

A target is the meta data relating to an observation. For example, it could be STX type, phage type, or region of origin for genome sequence datasets like ours.

Features
--------

The blanket term for what the context of the values represent in our dataset. For our genome sequence dataset, it is the counts of certain :math:`k`-mers and it is the context of the columns of a dataset.

Metrics
-------

We evaluate how well a model explains a dataset by examining a statistical metric. Common metrics are, Recall, Precision, and F1. See `Wikipedia <https://en.wikipedia.org/wiki/Precision_and_recall>`_ for some of these definitions.

k-mer 
-----

Sequencing fragments of length k, generated from splitting up assemblies or reads into smaller fragments.

unitig
------

High-confidence contigs of overlapping k-mers, generated from a search of a De Brujin graph for groups of overlapping fragments that together make a sequence that does not overlap with conflicting sequences.
