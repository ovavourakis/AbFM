"""
Script to pre-filter antibody data. Run on many CPU cores.

* use ImmuneBuilder dataset from https://zenodo.org/records/7258553
* merge with pre-defined train/val/test splits from https://zenodo.org/records/8164693
* filter out instances with inappropriate loci, intrusive stop codons, frameshifts
* reconstruct full sequences from ANARCI numbering
* keep instances where reconstructed sequence matches full sequence in entry
* keep instances with sequence length between MIN_SEQ_LEN and MAX_SEQ_LEN
* eliminate sequences with missing PDBs

"""
import pandas as pd
import os

from multiprocessing import Pool
from utils import ANARCIs_to_sequence

MIN_SEQ_LEN = 215
MAX_SEQ_LEN = 260
col_drop_filters = ['_aa_', '_alignment_', 'junction', 'cdr', 'fwr', 'fwk', 'fwl']

all_metadata_csv = "/vols/opig/users/vavourakis/data/OAS_models/OAS_paired_all.csv"
# all_metadata_csv = "/vols/opig/users/vavourakis/data/OAS_models/f20.csv"
splits_csv = "/vols/opig/users/vavourakis/data/oas_splits.csv"
strucs_dir = "/vols/opig/users/vavourakis/data/OAS_models/structures"
out_filtered_csv = "/vols/opig/users/vavourakis/data/OAS_models/OAS_paired_filtered.csv"

def process_row(x):
    return ANARCIs_to_sequence(x.ANARCI_numbering_heavy, x.ANARCI_numbering_light)


if __name__ == '__main__':

    print('\nreading metadata')
    df = pd.read_csv(all_metadata_csv)
    df.rename(columns={'full_seq': 'full_seq_orig'}, inplace=True)
    for filter in col_drop_filters:
        df.drop([col for col in df.columns if filter in col], axis=1, inplace=True)
    print('reading splits')
    splits = pd.read_csv(splits_csv)
    print('merging')
    df = pd.merge(df, splits, left_on='ID', right_on='oas_id', suffixes=('', '_drop'))
    df['oas_id'] = df['ID'].fillna(df['oas_id'])
    df.drop([col for col in df.columns if 'drop' in col], axis=1, inplace=True)

    print(f'\n{len(df)} entries before filtering')

    print('filtering instances with inappropriate loci, intrusive stop codons, frameshifts')
    df = df[(df['locus_heavy'] == 'H') & (df['locus_light'].isin(['K', 'L'])) &
            (df['stop_codon_heavy']=='F') & (df['stop_codon_light']=='F') &
            (df['vj_in_frame_heavy']=='T') & (df['vj_in_frame_light']=='T')]

    print(f'{len(df)} entries after filtering\n')

    print(f'wrangling sequences')
    with Pool(processes=24) as pool:
        results = pool.map(process_row, [row for _, row in df.iterrows()])
    df2 = pd.DataFrame(results, columns=['fwr1_h', 'cdr1_h', 'fwr2_h', 'cdr2_h', 'fwr3_h', 'cdr3_h', 'fwr4_h',
                                         'fwr1_l', 'cdr1_l', 'fwr2_l', 'cdr2_l', 'fwr3_l', 'cdr3_l', 'fwr4_l',
                                         'full_seq', 'cdr_concat', 'fw_concat', 'region_indices', 'seqlen']
                       )
    print(f'merging')
    df2.index = df.index
    df = pd.concat([df, df2], axis=1)
    del df2

    print(f'keeing instances where reconstructed sequence matches original full sequence')
    df = df[df['full_seq'] == df['full_seq_orig']]
    print(f'{len(df)} entries remaining\n')

    print(f'keeping sequences with length between {MIN_SEQ_LEN} and {MAX_SEQ_LEN}.')
    df = df[(df['seqlen'] >= MIN_SEQ_LEN) & (df['seqlen'] <= MAX_SEQ_LEN)]

    print(f'\n{len(df)} entries remaining\n')

    print(f'generating and checking pdb paths')
    df['pdb_path'] = df['oas_id'].apply(lambda x: os.path.join(strucs_dir, f'{x}.pdb'))
    file_mask = df['pdb_path'].apply(lambda path: os.path.exists(path))
    df = df[file_mask]

    print(f'\n{len(df)} entries remaining\n')

    cols_to_keep = ['oas_id', 'pdb_path', 'seqlen', 'cluster_ids', 'split', 'full_seq', 'locus_heavy', 'locus_light',
                    'fwr1_h', 'cdr1_h', 'fwr2_h', 'cdr2_h', 'fwr3_h', 'cdr3_h', 'fwr4_h',
                    'fwr1_l', 'cdr1_l', 'fwr2_l', 'cdr2_l', 'fwr3_l', 'cdr3_l', 'fwr4_l',
                    'cdr_concat', 'fw_concat', 'region_indices', 
                    'ANARCI_numbering_heavy', 'ANARCI_numbering_light']
    df = df[cols_to_keep]

    print(f'writing to {out_filtered_csv}')
    df.to_csv(out_filtered_csv, index=False)
    print('done')