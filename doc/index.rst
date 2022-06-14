##############
GenoLearn
##############

GenoLearn is a machine learning pipeline designed for biologists working with genome sequence data looking to build a predictive model or identify important patterns.

Installation
############

Pip
====================

Stable Install

Ubuntu / Mac OS
---------------

.. code-block:: sh

    # set a virtual environment if not already done so
    # assuming Python3 as the default python (otherwise use python3 inplace of python)
    python3 -m venv env

    # activate virtual environment
    source env/bin/activate

    # install GenoLearn
    pip install genolearn


Windows
-------

.. code-block:: sh

    # set a virtual environment if not already done so
    # assuming Python3 as the default python (otherwise use python3 inplace of python)
    python3 -m venv env

    # activate virtual environment
    ./env/Scripts/activate

    # install GenoLearn
    pip install genolearn


Conda
======================

Stable Install

.. code-block:: sh

    # create new conda environment
    conda create --name env

    # activate environment
    conda activate env

    # install GenoLearn
    conda install genolearn --channel jordan-wei-taylor


GitHub + Pip
======================================

Unstable Latest Install

This is to install the most recent work in progress. Note that some functionality may be be broken but once fully tested, the above installation methods will install the most recent stable version of ``genolearn``.


.. code-block:: sh

    pip install git+https://github.com/jordan-wei-taylor/genolearn.git

.. toctree::
    :hidden:
    :titlesonly:
    
    Installation <self>
    QuickStart <quickstart>
    glossary

.. toctree::
    :hidden:
    :caption: Usage

    usage/preprocess
    usage/feature-selection
    usage/train
    usage/inference

.. toctree::
    :hidden:
    :titlesonly:
    :caption: TBC

    tbc/overview
    tbc/preprocessing

.. toctree::
    :hidden:

    tbc/data-loader

.. toctree::
    :hidden:
    :titlesonly:
    
    tbc/feature-selection/index
    tbc/models/index
    tbc/feature-importance/index
    tbc/metrics
    
.. toctree::
    :hidden:
    :caption: Python Documentation

    genolearn/index

.. toctree::
    :hidden:
    :maxdepth: 4
    :caption: Case Studies
    :titlesonly:

    case-study/predicting-strain-origin-from-k-mer-counts/index
