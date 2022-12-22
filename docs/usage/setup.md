GenoLearn Setup
###############

Users need to setup the directory they want GenoLearn to work in. The user should open a terminal and ``cd`` into the directory they want to initialise their project in. The directory should contain their sequence data (.gz) and their metadata (comma seperated .csv or .txt).

To setup the current directory for GenoLearn to use, users should execute

.. code-block:: text

    genolearn-setup

which will prompt the user to select either the current directory or one of the available subdirectories as your ``data directory``  followed by selecting your ``metadata`` file. Upon successful execution, a ``.genolearn`` file is created in the current directory and users can now execute

.. code-block:: text

    genolearn

which then prints

.. code-block:: text

    Genolearn ({VERSION}) Command Line Interface

    GenoLearn is designed to enable researchers to perform Machine Learning on their genome
    sequence data such as fsm-lite or unitig files.

    See https://genolearn.readthedocs.io for documentation.

    Working directory: {WORKING_DIRECTORY} 

    1.  exit                               exits GenoLearn

    2.  print                              prints various GenoLearn generated files
    3.  preprocess                         preprocess data into an easier format for file reading

at which point, the user enters the **option number** to continue.

If users execute the above whilst not in the correct working directory, GenoLearn will try to access the last directory it was working in. If the last directory no longer exists or the ``.genolearn`` file has been deleted, GenoLearn will tell the user to first ``cd`` into a valid ``working directory`` or to execute ``genolearn-setup`` again.
