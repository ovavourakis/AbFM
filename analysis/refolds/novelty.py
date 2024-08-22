# TODO: do this for gen_strucs and also for trainset_orig

import os, argparse, glob
import pandas as pd
from SPACE2.util import reg_def
from SPACE2.exhaustive_clustering import get_distance_matrices

import matplotlib.pyplot as plt
import seaborn as sns


# parser = argparse.ArgumentParser(description='Calculate novelty rmsds for generated structures.')
# parser.add_argument('--gen_dir', type=str, required=True, help='Directory containing generated structures')
# args = parser.parse_args()

# read in best tm_score partners for each structure
# gen_dir = args.gen_dir
gen_dir = '/vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun'
csv_files = glob.glob(os.path.join(gen_dir, 'closest_tm_score_partners_in_trainset_*.csv'))
best_tm_partners = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# add all structures to a bucket and calculate pairwise RMSDs
all_partner_strucs = best_tm_partners['gen_struc'].tolist() + best_tm_partners['closest_train_struc'].tolist()
selection = [reg_def['CDRH3']]
out = get_distance_matrices(all_partner_strucs, selection=selection, anchors=selection, n_jobs=-1)

for i, row in best_tm_partners.iterrows():
    gen_struc = row['gen_struc']
    closest_train_struc = row['closest_train_struc']
    for cdr_len, (strucs, matrix) in out.items():
        if gen_struc in strucs:
            gen_struc_index = strucs.index(gen_struc)
            closest_train_struc_index = strucs.index(closest_train_struc)
            rmsd = matrix[gen_struc_index, closest_train_struc_index]
            condition = (best_tm_partners['gen_struc'] == gen_struc) & \
                        (best_tm_partners['closest_train_struc'] == closest_train_struc)
            best_tm_partners.loc[condition, 'cdrh3_rmsd'] = rmsd

print(f'% generated structures with unique trainset partner: {best_tm_partners['closest_train_struc'].nunique()/len(best_tm_partners)*100}')

plt.figure(figsize=(6, 10), dpi=300)
sns.violinplot(y=best_tm_partners['cdrh3_rmsd'].dropna(), palette="Set3", cut=0, legend=False)
plt.ylabel('CDRH3 RMSD (Angstrom)')
plt.title('CDRH3 RMSD to Closest Trainset \nMatch by TM-Score', fontsize=16)
plt.xticks([0], ['Generated'])
outfile = os.path.join(gen_dir, 'designed_seqs', 'tm_cdrh3_rmsd_violinplot.png')
plt.tight_layout()
plt.savefig(outfile)
plt.show()
