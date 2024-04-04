"""Script to filter Antibody Data."""
import pandas as pd
import os

MIN_SEQ_LEN = 215
MAX_SEQ_LEN = 260

strucs_dir = "/vols/opig/users/vavourakis/data/OAS_models/structures"
all_metadata_csv = "/vols/opig/users/vavourakis/data/OAS_models/OAS_paired_all.csv"
splits_csv = "/vols/opig/users/vavourakis/data/oas_splits.csv"

out_filtered_csv = "/vols/opig/users/vavourakis/data/OAS_models/OAS_paired_filtered.csv"
out_minim_csv = "/vols/opig/users/vavourakis/data/OAS_models/OAS_filtered_minimal.csv"


print('reading metadata')
df = pd.read_csv(all_metadata_csv)
print('reading splits')
splits = pd.read_csv(splits_csv)

print('merging')
df = pd.merge(df, splits, left_on='ID', right_on='oas_id', suffixes=('', '_drop'))
df['oas_id'] = df['ID'].fillna(df['oas_id'])
df.drop([col for col in df.columns if 'drop' in col], axis=1, inplace=True)

print('filtering instances with inappropriate loci')
df = df[df['locus_heavy'] == 'H']
df = df[df['locus_light'].isin(['K', 'L'])]

print(f'keeping sequences with length between {MIN_SEQ_LEN} and {MAX_SEQ_LEN}')
df['seqlen'] = df['full_seq'].str.len()
df = df[(df['seqlen'] >= MIN_SEQ_LEN) & (df['seqlen'] <= MAX_SEQ_LEN)]

print('concatenating CDRs')
cdr_aa_heavy = df['cdr1_aa_heavy'] + df['cdr2_aa_heavy'] + df['cdr3_aa_heavy']
cdr_aa_light = df['cdr1_aa_light'] + df['cdr2_aa_light'] + df['cdr3_aa_light']
df['concatCDR_aa'] = cdr_aa_heavy + cdr_aa_light


print('generating pdb paths')
df['pdb_path'] = df['oas_id'].apply(lambda x: os.path.join(strucs_dir, f'{x}.pdb'))

print('writing out full dataset')
df.to_csv(out_filtered_csv, index=False)

cols_to_keep = ['oas_id', 'pdb_path', 'seqlen', 'cluster_id', 'split', 'full_seq', 'sequence_heavy', 'sequence_light', 'locus_heavy', 'locus_light',
                'cdr1_start_heavy', 'cdr1_end_heavy', 'cdr2_start_heavy', 'cdr2_end_heavy', 'cdr3_start_heavy', 'cdr3_end_heavy',
                'cdr1_start_light', 'cdr1_end_light', 'cdr2_start_light', 'cdr2_end_light', 'cdr3_start_light', 'cdr3_end_light',
                'cdr1_aa_heavy', 'cdr2_aa_heavy', 'cdr3_aa_heavy', 'cdr1_aa_light', 'cdr2_aa_light', 'cdr3_aa_light', 
                'concatCDR_aa', 
                'ANARCI_numbering_heavy', 'ANARCI_numbering_light']