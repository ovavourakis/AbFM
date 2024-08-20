import os, argparse
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from analysis.refolds.utils import numbering_to_region_index, get_backbone_coordinates, superpose_strucs, get_rmsd

parser = argparse.ArgumentParser(description='Compute RMSDs between scaffolds and their refolds.')
parser.add_argument('--recompute_global', action='store_true', help='Recompute global RMSDs even if they already exist.')
parser.add_argument('--recompute_local', action='store_true', help='Recompute local RMSDs even if they already exist.')
args = parser.parse_args()

def compute_global_rmsd(coords1, coords2):
    sup, _ = superpose_strucs(coords1, coords2)
    return sup.get_rms()

def check_consistent_numbering(anarci_df, use_north=True):
    """Check if all sequences for each de novo structure have consistent numbering."""
    inconsistent_structures = []
    for (structure, chain), group in anarci_df.groupby(['structure', 'chain']):
        region_dicts = group.apply(lambda row: numbering_to_region_index(row['domain_numbering'], row['seq'], row['chain'], use_north=use_north), axis=1)
        first_dict = region_dicts.iloc[0]
        if not all(region_dict == first_dict for region_dict in region_dicts):
            inconsistent_structures.append((structure, chain))
    return inconsistent_structures

def construct_paths(anarci_df, gen_dir, type='denovo'):
    paths = set()
    for _, row in anarci_df.iterrows():
        structure = row['structure']
        if type == 'denovo':
            parts = structure.split('_')
            totlen, sample = "_".join(parts[0:2]), "_".join(parts[2:])
            scaffold_path = f"{gen_dir}/{totlen}/{sample}/sample.pdb"
            refolded_paths = [f"{gen_dir}/refolded_strucs/{totlen}_{sample}_seq_{i}_fold.pdb" for i in range(20)]
        elif type == 'reference':
            scaffold_path = f"/vols/opig/users/vavourakis/data/new_OAS_models/structures/{structure}.pdb"
            refolded_paths = [f"{gen_dir}/refolded_strucs/{structure}_seq_{i}_fold.pdb" for i in range(20)]

        for path in [scaffold_path] + refolded_paths:
            assert os.path.exists(path), f"Path does not exist: {path}"
        paths.add((scaffold_path, tuple(refolded_paths)))
    return paths

def scaffold_refolds_global_rmsd(anarci_df, gen_dir, type='denovo'):
    paths = construct_paths(anarci_df, gen_dir, type=type)
    
    all_results, best_rows = [], []
    for scaffold_path, refolded_paths in tqdm(paths):
        scaffold_coords = get_backbone_coordinates(scaffold_path)
        results = [{'scaffold_path': scaffold_path, 'refolded_path': refolded_path, 'global_rmsd': compute_global_rmsd(scaffold_coords, get_backbone_coordinates(refolded_path))} for refolded_path in refolded_paths]
        best_row = min(results, key=lambda x: x['global_rmsd'])
        all_results.extend(results)
        best_rows.append(best_row)

    return pd.DataFrame(all_results), pd.DataFrame(best_rows)

def scaffold_refold_region_rmsd(best_rmsd_df, anarci_df, use_north=True, type='denovo'):
    rmsds_per_region = []
    for _, row in tqdm(best_rmsd_df.iterrows()):

        # superpose structures globally
        scaffold_coords = get_backbone_coordinates(row['scaffold_path'])
        best_refold_coords = get_backbone_coordinates(row['refolded_path'])
        assert scaffold_coords.shape == best_refold_coords.shape, f"Shape mismatch: {scaffold_coords.shape} != {best_refold_coords.shape}"
        sup, transformed_refold_coords = superpose_strucs(scaffold_coords, best_refold_coords)
        
        # get structure and sequence id (to match up ANARCI numberings)
        parts = row['refolded_path'].split('/')[-1].split('.')[0].split('_')
        if type == 'denovo':
            structure, sequence_id = '_'.join(parts[0:4]), int(parts[-2])
        elif type == 'reference':
            structure, sequence_id = parts[0], int(parts[-2])

        # compute rmsd per region
        vh_len, regional_rmsds = None, {}
        for chain in ['H', 'L']:
            condition = (anarci_df['structure'] == structure) & (anarci_df['seq_id'] == sequence_id) & (anarci_df['chain'] == chain)
            anarci_string = anarci_df[condition]['domain_numbering'].iloc[0]
            sequence = anarci_df[condition]['seq'].iloc[0]
            
            if chain == 'H':
                vh_len = len(sequence)
            else:
                assert (vh_len+len(sequence))*4 == scaffold_coords.shape[0] == transformed_refold_coords.shape[0], f"Sequence length mismatch: {len(sequence)} != {scaffold_coords.shape[0]} != {transformed_refold_coords.shape[0]}"

            region_to_index = numbering_to_region_index(anarci_string, sequence, chain, use_north=use_north) # maps region -> sequential sequence index

            # get rmsd in globally aligned reference frame
            for region, seq_indices in region_to_index.items():
            
                indices = [j for i in seq_indices for j in range(i*4, i*4+4)]   # four atoms per residue
                if chain == 'L': 
                    indices = [i + vh_len*4 for i in indices]                   # also add the length of the VH region
                    
                key = f'{region}' if use_north else f'{region}_{chain}'
                regional_rmsds[key] =  get_rmsd(scaffold_coords[indices], transformed_refold_coords[indices])
        rmsds_per_region.append(regional_rmsds)

    return pd.concat([best_rmsd_df, pd.DataFrame(rmsds_per_region)], axis=1)

gen_dir = '/vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun'
ref_dir = '/vols/opig/users/vavourakis/generations/TRAINSET_genseq2'

# read in anarci data for region definitions
anarci_df = pd.read_csv(os.path.join(gen_dir, 'designed_seqs/anarci_annotation.csv'))
ref_anarci_df = pd.read_csv(os.path.join(ref_dir, 'anarci_annotation.csv'))

if args.recompute_global:
    # check if all sequences for each de novo structure have consistent numbering
    print('Checking for consistent numbering...')
    inconsistent_structures = check_consistent_numbering(anarci_df, use_north=True)
    if inconsistent_structures:
        print(f"WARNING! Some scaffolds have sequence sets with inhomogenous numberings: {inconsistent_structures}\n")

    # compute global RMSDs between scaffold and refolded structures
    print('Calculating global RMSDs (REFERENCE SET)...')
    all_ref_rmsd_df, best_ref_rmsd_df = scaffold_refolds_global_rmsd(ref_anarci_df, ref_dir, type='reference')
    all_ref_rmsd_df.to_csv(os.path.join(ref_dir, 'global_rmsds.csv'), index=False)
    best_ref_rmsd_df.to_csv(os.path.join(ref_dir, 'best_global_rmsds.csv'), index=False)
    print('Calculating global RMSDs (GENERATED SET)...')
    all_rmsd_df, best_rmsd_df = scaffold_refolds_global_rmsd(anarci_df, gen_dir, type='denovo')
    all_rmsd_df.to_csv(os.path.join(gen_dir, 'designed_seqs/global_rmsds.csv'), index=False)
    best_rmsd_df.to_csv(os.path.join(gen_dir, 'designed_seqs/best_global_rmsds.csv'), index=False)
else:
    all_ref_rmsd_df = pd.read_csv(os.path.join(ref_dir, 'global_rmsds.csv'))
    best_ref_rmsd_df = pd.read_csv(os.path.join(ref_dir, 'best_global_rmsds.csv'))
    all_rmsd_df = pd.read_csv(os.path.join(gen_dir, 'designed_seqs/global_rmsds.csv'))
    best_rmsd_df = pd.read_csv(os.path.join(gen_dir, 'designed_seqs/best_global_rmsds.csv'))

if args.recompute_local:
    # compute region-wise RMSDs between each scaffold and *best* refolded structure
    print('Calculating local RMSDs (REFERENCE SET)...')
    ref_best_rmsds = scaffold_refold_region_rmsd(best_ref_rmsd_df, ref_anarci_df, use_north=True, type='reference')
    ref_best_rmsds.to_csv(os.path.join(ref_dir, 'best_rmsds.csv'), index=False)
    print('Calculating local RMSDs (GENERATED SET)...')
    best_rmsds = scaffold_refold_region_rmsd(best_rmsd_df, anarci_df, use_north=True, type='denovo')
    best_rmsds.to_csv(os.path.join(gen_dir, 'designed_seqs/best_rmsds.csv'), index=False)
else:
    best_rmsds = pd.read_csv(os.path.join(gen_dir, 'designed_seqs/best_rmsds.csv'))
    ref_best_rmsds = pd.read_csv(os.path.join(ref_dir, 'best_rmsds.csv'))



print('REFERENCE SET:')
print(ref_best_rmsds.describe())
print('\n')
ref_to_plot = ref_best_rmsds.drop(columns=['scaffold_path', 'refolded_path'])
ref_fully_designable = (ref_to_plot < 2).all(axis=1).mean()
ref_globally_designable = (ref_to_plot['global_rmsd'] < 2).mean()
print(f"ALL REGIONS DESIGNABLE: {ref_fully_designable*100}%")
print(f"GLOBALLY DESIGNABLE: {ref_globally_designable*100}%")

print('\nDE NOVO SET:')
print(best_rmsds.describe())
print('\n')
to_plot = best_rmsds.drop(columns=['scaffold_path', 'refolded_path'])
fully_designable = (to_plot < 2).all(axis=1).mean()
globally_designable = (to_plot['global_rmsd'] < 2).mean()
print(f"ALL REGIONS DESIGNABLE: {fully_designable*100}%")
print(f"GLOBALLY DESIGNABLE: {globally_designable*100}%")



print('Plotting results...')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

palette = {'global_rmsd': 'green', 'FR1_H': 'lightblue', 'CDR1_H': 'blue', 'FR2_H': 'lightblue', 'CDR2_H': 'blue', 
           'FR3_H': 'lightblue', 'CDR3_H': 'blue', 'FR4_H': 'lightblue', 'FR1_L': 'lightcoral', 'CDR1_L': 'red', 
           'FR2_L': 'lightcoral', 'CDR2_L': 'red', 'FR3_L': 'lightcoral', 'CDR3_L': 'red', 'FR4_L': 'lightcoral'}

y_max, y_min = max(to_plot.max().max(), ref_to_plot.max().max())*1.05, min(to_plot.min().min(), ref_to_plot.min().min())*0.95

def plot_boxplot(ax, data, title, textstr):
    sns.boxplot(ax=ax, data=data, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"}, palette=palette)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('scRMSD (Angstrom)')
    ax.axvline(x=0.5, color='gray', linestyle='--')
    ax.axvline(x=7.5, color='gray', linestyle='--')
    ax.axhline(y=2, color='red', linestyle='--')
    ax.text(1, 2.05, 'designability cutoff', color='red', verticalalignment='bottom', horizontalalignment='left')
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(['Global', 'FRH1', 'CDRH1', 'FRH2', 'CDRH2', 'FRH3', 'CDRH3', 'FRH4', 'FRL1', 'CDRL1', 'FRL2', 'CDRL2', 'FRL3', 'CDRL3', 'FRL4'], rotation=45)
    ax.set_ylim(y_min, y_max)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plot_boxplot(axes[0], to_plot, 'De Novo', f'Globally Designable: {globally_designable*100:.2f}%\n All Regions Designable: {fully_designable*100:.2f}%')
plot_boxplot(axes[1], ref_to_plot, 'Training Set', f'Globally Designable: {ref_globally_designable*100:.2f}%\n All Regions Designable: {ref_fully_designable*100:.2f}%')

fig.suptitle('scRMSDs by North Region for Best Refolded Structure among 20', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(gen_dir, 'designed_seqs/best_rmsds_boxplot.png'))
plt.show()