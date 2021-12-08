from biolearn.process import process, process_groupby
from biolearn.logger  import msg, print_dict

if __name__ == '__main__':

    description = \
    r"""
    Processes a gunzip (gz) compressed text file of the following sparse format

    feature_id_1 | sample_id_1:value_1_1 sample_id_2:value_1_2 ...\n
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
        python -m kmer_ml.scripts.process2dense raw-data/STEC_14-19_fsm_kmers.txt.gz raw-data/meta_data.csv Accession data Region --groupby Year 
    """

    from   argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(description = description, formatter_class = RawTextHelpFormatter)

    parser.add_argument('input_file')
    parser.add_argument('meta_path')
    parser.add_argument('identifier')
    parser.add_argument('output_directory')
    parser.add_argument('target', nargs='+')
    parser.add_argument('--groupby', default = False)

    args   = parser.parse_args()
    params = dict(args._get_kwargs())
    print_dict('executing "process.py" with parameters:', params)

    if args.groupby:
        process_groupby(args.input_file, args.meta_path, args.identifier, args.groupby, args.output_directory, args.target)
    else:
        process(args.input_file, args.meta_path, args.identifier, args.output_directory, args.target)
    
    msg('executed "process.py"')
