import os
import pandas as pd
from tqdm import tqdm

from analysis.refolds.utils import numbering_to_region_index, get_backbone_coordinates, superpose_strucs, get_rmsd

def construct_paths(anarci_df, gen_dir):
    paths = []
    for _, row in anarci_df.iterrows():
        structure = row['structure']
        parts = structure.split('_')
        totlen, sample = "_".join(parts[0:2]), "_".join(parts[2:])
        scaffold_path = f"{gen_dir}/{totlen}/{sample}/sample.pdb"
        assert os.path.exists(scaffold_path), f"Path does not exist: {scaffold_path}"
        paths.append(scaffold_path)
    return paths

def get_cdr_lengths(anarci_df):
    cdr_lengths = []
    for _, row in tqdm(anarci_df.iterrows()):
        anarci_string = row.domain_numbering
        sequence = row.seq
        region_to_index = numbering_to_region_index(anarci_string, sequence, row.chain)
        cdr_lengths.append({region: len(index_list) for region, index_list in region_to_index.items() if 'CDR' in region})
    return pd.DataFrame(cdr_lengths)

def find_corresponding_vh_lens(anarci_df, row1, row2):
    # TODO: implement this
    return 0, 0 

def compute_pairwise_rmsds_for_cdr_type(cdr_df, cdr_type, anarci_df):
    """Requires cdr_df of structures length-homogenous in cdr of cdr_type."""
    pwd_rmsds = []
    for i, row1 in cdr_df.iterrows():
        for j, row2 in cdr_df.iterrows():
            if i >= j:
                continue

            # get sequence indices of current cdr
            seq_indices1 = numbering_to_region_index(row1['domain_numbering'], row1['seq'], row1['chain'])
            seq_indices2 = numbering_to_region_index(row2['domain_numbering'], row2['seq'], row2['chain'])
            
            # convert that to indices in coordinate tensors
            vh_len_1, vh_len_2 = find_corresponding_vh_lens(anarci_df, row1, row2)
            indices1 = [j + (vh_len_1*4 if row1['chain'] == 'L' else 0) for i in seq_indices1[cdr_type] for j in range(i*4, i*4+4)]
            indices2 = [j + (vh_len_2*4 if row1['chain'] == 'L' else 0) for i in seq_indices2[cdr_type] for j in range(i*4, i*4+4)]

            # get full structures and superpose
            coords1 = get_backbone_coordinates(row1['pdb_file'])
            coords2 = get_backbone_coordinates(row2['pdb_file'])
            sup, new_coords2 = superpose_strucs(coords1, coords2)

            # compute rmsd of this cdr without re-aligning sub-structures
            rmsd = get_rmsd(coords1[indices1], coords2[indices2])
            pwd_rmsds.append(rmsd)
    return pwd_rmsds

# read in anarci data for region definitions
gen_dir = '/vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun'
anarci_df = pd.read_csv(os.path.join(gen_dir, 'designed_seqs/anarci_annotation.csv'))
anarci_df['pdb_file'] = construct_paths(anarci_df, gen_dir)
print('Measuring CDR lengths...')
anarci_df = pd.concat([anarci_df, get_cdr_lengths(anarci_df)], axis=1)

# iterate through cdrs, match up structures of identical cdr length, compute all pairwise rmsds
cdrs = ['CDR1_H', 'CDR2_H', 'CDR3_H', 'CDR1_L', 'CDR2_L', 'CDR3_L']
pwd_rmsds = {cdr: [] for cdr in cdrs}
for cdr_type in cdrs:
    cdr_df = anarci_df[anarci_df[cdr_type] > 0] # filter out NaN-lengths from other chain type
    for cdr_len, cdr_df_group in cdr_df.groupby(cdr_type):
        pwd_rmsds[cdr_type].extend(compute_pairwise_rmsds_for_cdr_type(cdr_df_group, cdr_type, anarci_df))


# TODO: how to superpose structures of different lengths with equal-length cdrs?
# TODO: fix the find_vh_lens function
# TODO: violinplot of cdr rmsd distributions for each cdr type