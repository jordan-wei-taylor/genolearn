Preprocessing
#############

``GenoLearn's`` preprocessing script assumes you have data in the following form

.. code-block:: text

    sequence_1 | identifier_{1,1}:count_{1,1} identifier_{1,1}:count_{2,1} ...
    sequence_2 | identifier_{2,1}:count_{2,1} identifier_{2,1}:count_{2,2} ...
    ...

Your data is expected to have a file extention of ``.txt`` or ``.gz`` in the case of compressed formatting using ``gunzip``. To preprocess your dataset, we first need to define an output directory and the path to the genome sequence data. As an example, we have a file that consists of various strains of E. Coli O157 collected over the period 2014-2018 wih path ``raw-data/STEC_14-18_fsm_kmers.txt.gz``.

.. code-block:: bash

    # define the output directory and the input genome sequence path
    output="data"
    sequence="raw-data/STEC_14-18_fsm_kmers.txt.gz"

    # execute the preprocessing script with default optional parameters
    python3 -m genolearn.preprocess $output $sequence

The resulting script should output the following directory

.. code-block:: text
    
    data
    ├── dense
    |   ├── *.npz
    ├── feature-selection
    ├── sparse
    |   ├── *.npz
    ├── features.7z
    ├── log.txt
    └── meta.json

Combining Genome Sequence Datasets
==================================

Suppose we collect more genome sequence data in the year 2019 with the following path ``raw-data/STEC_19_fsm_kmers.txt.gz`` and we would like to combine it's preprocessing contents with the previously generated ``data`` directory relating to ``raw-data/STEC_14-18_fsm_kmers.txt.gz``, we can call a similar script:

.. code-block:: bash

    # define the output directory and the input genome sequence path
    output="data"
    sequence="raw-data/STEC_19_fsm_kmers.txt.gz"

    python3 -m genolearn.combine $output $sequence

The above command will populate more entries into the sub-directories ``data/dense`` and ``data/sparse`` if those directories exist. See :ref:`Preprocessing` for more details detailing how these sub-directories are used.
