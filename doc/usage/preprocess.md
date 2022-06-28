Preprocess
##########

Creates a preprocessed data directory given a genome sequence file (compressed with gunzip).

Default preprocessing usage

.. code-block:: bash

    output_dir="data"
    genome_sequence_path="genome_data.txt.gz"

    python3 -m genolearn $output_dir $genome_sequence_path

Executing the above would create both dense and sparse representations of your data which can be gigabytes of space. It is more often the case that you do not require sparse representation so we can ommit this output by appending ``--sparse false`` i.e.

.. code-block:: bash

    python3 -m genolearn $output_dir $genome_sequence_path --sparse false

The preprocessing of the data relies on either generating temporary txt files or storing this temporary data in memory. Writing these temporary files increases the preprocessing time whilst storing them in memory inreases the required memory allowance. If you have extra memory to spare (RAM), we can choose to not output any temporary files by appending the ``--not-low-memory``. If you have harddisk to spare, you can try increasing the ``batch_size``. Either option should decrease preprocessing time significantly as long as your computer can handle either the increase RAM usage or the additional harddisk space.

By default, the verbose updates occur every 250,000 sequences. If the user prefers a different update number, say 100,000, they can change it by appending ``verbose 100000``. Similarly, the default number of parallel processes to run is the number of CPU cores Python can see.

The default options are as follows:

.. list-table:: Default Parameters for GenoLearn's Preprocessing
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Parameter
     - Flag
     - Default Value
   * - batch size
     - \-\-batch_size
     - 512
   * - verbose
     - \-\-verbose
     - 250000
   * - no. processes
     - \-\-n_processes
     - auto
   * - sparse
     - \-\-sparse
     - true
   * - dense
     - \-\-dense
     - true
   * - debug
     - \-\-debug
     - -1
   * - not low memory
     - \-\-not-low-memory
     - false