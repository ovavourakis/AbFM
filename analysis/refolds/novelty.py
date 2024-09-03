import os, argparse, glob
import pandas as pd
from SPACE2.util import reg_def
from SPACE2.exhaustive_clustering import get_distance_matrices

import matplotlib.pyplot as plt
import seaborn as sns

# parser = argparse.ArgumentParser(description='Calculate novelty rmsds for generated structures.')
# parser.add_argument('--gen_dir', type=str, required=True, help='Directory containing generated structures')
# parser.add_argument('--ref_dir1', type=int, required=True, help='Directory containing comparable reference set structures')
# parser.add_argument('--ref_dir2', type=int, required=True, help='Directory containing comparable reference set structures')
# args = parser.parse_args()

# read in best tm_score partners for each structure
# gen_dir = args.gen_dir
# ref_dir1 = args.ref_dir1
# ref_dir2 = args.ref_dir2
gen_dir = '/vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun'
ref_dir1 = '/vols/opig/users/vavourakis/generations/TRAINSET_origseq3'
ref_dir2 = '/vols/opig/users/vavourakis/generations/TRAINSET_origseq4'
gen_csv_files = glob.glob(os.path.join(gen_dir, 'closest_tm_score_partners_in_trainset_*.csv'))
ref1_csv_files = glob.glob(os.path.join(ref_dir1, 'closest_tm_score_partners_in_trainset_*.csv'))
ref2_csv_files = glob.glob(os.path.join(ref_dir2, 'closest_tm_score_partners_in_trainset_*.csv'))

best_tm_partners = None
best_tm_partners_ref1 = None
best_tm_partners_ref2 = None
for dir_path, csv_files, output_var in [(gen_dir, gen_csv_files, 'best_tm_partners'), 
                                        (ref_dir1, ref1_csv_files, 'best_tm_partners_ref1'),
                                        (ref_dir2, ref2_csv_files, 'best_tm_partners_ref2')]:
    if not csv_files:
        result = pd.read_csv(os.path.join(dir_path, 'best_tm_partner_rmsds.csv'))
    else:
        result = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        # add all structures to a bucket and calculate pairwise RMSDs
        all_partner_strucs = result['gen_struc'].tolist() + result['closest_train_struc'].tolist()
        selection = [reg_def['CDRH3']]
        out = get_distance_matrices(all_partner_strucs, selection=selection, anchors=selection, n_jobs=-1)

        for i, row in result.iterrows():
            gen_struc = row['gen_struc']
            closest_train_struc = row['closest_train_struc']
            for cdr_len, (strucs, matrix) in out.items():
                if gen_struc in strucs:
                    gen_struc_index = strucs.index(gen_struc)
                    closest_train_struc_index = strucs.index(closest_train_struc)
                    rmsd = matrix[gen_struc_index, closest_train_struc_index]
                    condition = (result['gen_struc'] == gen_struc) & \
                                (result['closest_train_struc'] == closest_train_struc)
                    result.loc[condition, 'cdrh3_rmsd'] = rmsd
        result.to_csv(os.path.join(dir_path, 'best_tm_partner_rmsds.csv'), index=False)
    
    if output_var == 'best_tm_partners':
        best_tm_partners = result
    elif output_var == 'best_tm_partners_ref1':
        best_tm_partners_ref1 = result
    elif output_var == 'best_tm_partners_ref2':
        best_tm_partners_ref2 = result

plt.figure(figsize=(7.5, 6), dpi=300)
combined_data = pd.concat([best_tm_partners[['cdrh3_rmsd']].dropna().assign(Source='Generated'), 
                           best_tm_partners_ref1[['cdrh3_rmsd']].dropna().assign(Source='Length-Combo Reference'),
                           best_tm_partners_ref2[['cdrh3_rmsd']].dropna().assign(Source='CDRH3-Length Reference')])
sns.violinplot(x='Source', y='cdrh3_rmsd', data=combined_data, palette="Set3", cut=0, legend=False)
plt.ylabel('CDRH3 RMSD (Angstrom)')
plt.title('CDRH3 RMSD to Closest Same-Length Trainset Match by TM', fontsize=16)

plt.xticks([0, 1, 2], ['Generated', 'Transet (Matched for\n Chain-Length Combinations)', 'Trainset (Matched for\n CDRH3 Lengths)'])
plt.xlabel('')  # remove the x-axis label 'Source'
outfile = os.path.join(gen_dir, 'designed_seqs', 'tm_cdrh3_rmsd_violinplot.png')
plt.tight_layout()
plt.savefig(outfile)

# print('Cleaning up...')
# [os.remove(csv_file) for csv_file in csv_files+ref1_csv_files+ref2_csv_files]