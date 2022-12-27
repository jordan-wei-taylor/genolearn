##############
GenoLearn
##############

GenoLearn is a machine learning pipeline designed for biologists working with genome sequence data looking to build a predictive model or identify important patterns.

Installation
############

Ensure you have a minimum Python3 version of 3.10.

Pip
====================

Latest Stable Install

Ubuntu / Mac OS
---------------

.. code-block:: sh

    # set a virtual environment if not already done so
    python3 -m venv env

    # activate virtual environment
    source env/bin/activate

    # install GenoLearn
    pip install genolearn


Windows
-------

.. code-block:: sh

    # set a virtual environment if not already done so
    python3 -m venv env

    # activate virtual environment
    ./env/Scripts/activate

    # install GenoLearn
    pip install genolearn


Latest Unstable Development Install

.. code-block:: sh
    
    pip install -e git+https://github.com/jordan-wei-taylor/genolearn.git#egg=genolearn


Conda
======================

Stable Install

.. code-block:: sh

    # make sure my channel and conda-forge are your first channels for installation
    conda config --prepend channels conda-forge --prepend channels jordan-wei-taylor

    # create a new conda environment env
    conda create --name env

    # activate environment to use
    conda activate env

    # install genolearn
    conda install genolearn


Unstable Development Install

.. code-block:: sh
    
    conda install genolearn-dev



.. toctree::
    :hidden:
    :titlesonly:
    
    Installation <self>
    tutorial
    glossary
    
.. toctree::
    :hidden:
    :titlesonly:
    :caption: Background

    Fisher Score <background/fisher-score>
    Metrics <background/metrics>

.. toctree::
    :hidden:
    :titlesonly:
    :caption: Usage

    Overview <usage/overview>
    Setup <usage/setup>
    Clean <usage/clean>
    Print <usage/print>
    Preprocess <usage/preprocess>
    Feature Selection <usage/feature-selection>
    Model Config <usage/model-config>
    Train <usage/train>
    Feature Importance <usage/feature-importance>
    Evaluate <usage/evaluate>

.. toctree::
    :hidden:
    :maxdepth: 4
    :caption: Case Studies
    :titlesonly:

    case-study/predicting-strain-origin-from-k-mer-counts/index
