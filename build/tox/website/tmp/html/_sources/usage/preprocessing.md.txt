.. _Preprocessing:

Preprocessing
##########################


.. code-block:: text

    >>> python -m biolearn --help

    usage: __main__.py [-h] [--groupby GROUPBY] input_file meta_path identifier output_directory target [target ...]

        Processes a gunzip (gz) compressed text file of the following sparse format

        feature_id_1 | sample_id_1:value_1_1 sample_id_2:value_1_2 ...
        feature_id_2 | ...

        into a gunzip compressed text file which contains a matrix. The ij-th element of the matrix refers to the value at the 
        i-th feature and j-th sample i.e. value_i_j at feature_id_i, sample_id_j.

        Required Arguments
        =======================
            input_path         : path to compressed text file with sparse format
            meta_path          : path to csv containing identifiers and meta information e.g. "Region" or "Year"
            identifier         : name of column in meta csv containing all sample ids
            output_directory   : directory to output all generated files
            targets            : name of column(s) in meta csv containing desired target output(s)
            groupby [optional] : name of column in meta csv to group outputs by (see example)

        Example Usage
        =======================
            >>> python -m biolearn raw-data/STEC_14-19_fsm_kmers.txt.gz raw-data/meta_data.csv Accession data Region --groupby Year 
        

    positional arguments:
    input_file
    meta_path
    identifier
    output_directory
    target

    optional arguments:
    -h, --help         show this help message and exit
    --groupby GROUPBY
