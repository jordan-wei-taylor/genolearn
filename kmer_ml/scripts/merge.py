if __name__ == '__main__':
    import pandas as pd
    import numpy  as np

    df1 = pd.read_csv('raw-data/Foreign_travel_metadta_14_18.txt.gz', sep = '\t')
    df2 = pd.read_csv('raw-data/Foreign_travel_metadta_19.txt.gz', sep = '\t', header = None)
    Y   = pd.read_csv('raw-data/STEC_accession_year', sep = '\t')

    df  = pd.DataFrame(np.vstack([df1.values, df2.values]), columns = df1.columns)
    y   = {SRR : year for year, SRR in zip(*Y.values.T)}

    for SRR in df['Accession'].values.copy():
        row = df['Accession'] == SRR
        if SRR in y:
            df.loc[row, 'Year'] = y[SRR]
        else:
            df = df.loc[~row]

    df['Year'] = df['Year'].apply(int)

    df.to_csv('meta_data.csv', index = False, header = True, sep = '\t')
