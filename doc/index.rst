##############
GenoLearn
##############

GenoLearn is a machine learning pipeline designed for biologists working with fsm-lite files looking to build a predictive model or identify important patterns.

Installation
############

Pip
====================


.. code-block:: text

    # set a virtual environment if not already done so
    # assuming Python3 as the default python (otherwise use python3 inplace of python)
    >>> python -m venv env

    # activate virtual environment
    # if linux / mac
    >>> source env/bin/activate

    # if windows
    >>> ./env/Scripts/activate

    # install GenoLearn
    >>> pip install -U genolearn
    

GitHub
=======================

For the most recent stable version of GenoLearn clone the directory and then pip install

.. code-block:: text

    # clones repository to current directory
    >>> git clone https://github.com/jordan-wei-taylor/genolearn.git

    # change directory to the cloned repository
    >>> cd genolearn

    # install GenoLearn from local files
    >>> pip install -e .

Conda
======================

.. code-block:: text

    # create new conda environment
    >>> conda create --name env

    # activate environment
    >>> conda activate env

    # install GenoLearn
    >>> conda install genolearn --channel jordan-wei-taylor

.. toctree::
    :hidden:
    
    Home <self>

.. toctree::
    :hidden:
    :maxdepth: 4
    :titlesonly:
    :caption: Usage

    usage/overview
    usage/preprocessing
    usage/data-loader/index
    usage/feature-selection/index
    usage/models/index
    usage/feature-importance/index
    usage/metrics

.. toctree::
    :hidden:
    :caption: Python Documentation

    genolearn/index

.. toctree::
    :hidden:
    :maxdepth: 4
    :caption: Demos
    :titlesonly:

    demos/paper-a
    demos/paper-b