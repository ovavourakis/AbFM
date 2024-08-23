"""
Calculates the TM-score of a generated structure against the training set and saves the best-matching partner to csv.
To be called from batch script.

Usage:
    python TM_score.py --jobindex 0 --gen_dir /vols/opig/users/vavourakis/generations/TRAINSET_origseq3
"""

import glob, os, subprocess, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from SPACE2.util import reg_def, parse_antibodies, cluster_antibodies_by_CDR_length

import matplotlib.pyplot as plt
import seaborn as sns

def get_TMscore(pdb_query, pdb_reference):
    # TM score normalised by length of reference structure
    binary_path = '/vols/opig/users/vavourakis/codebase/US-align/USalign'
    command = f"{binary_path} {pdb_query} {pdb_reference} -mm 1 -ter 1 -outfmt 2 | awk 'NR==2 {{print $3}}'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    tm_score = float(result.stdout.strip())
    return tm_score

parser = argparse.ArgumentParser(description='TM-score each generated structure against training set.')
parser.add_argument('--jobindex', type=int, required=True, help='Index of the job to process')
parser.add_argument('--gen_dir', type=str, required=True, help='Directory containing generated structures')
args = parser.parse_args()

# define IMGT-numbered pdb files
print('Loading data...')
gen_dir = args.gen_dir
if os.path.exists(os.path.join(gen_dir, 'renumbered_pdbs')):
    pdb_files = glob.glob(os.path.join(gen_dir, 'renumbered_pdbs', '*.pdb'))
else:
    pdb_files = glob.glob(os.path.join(gen_dir, 'strucs', '*.pdb'))

train_dir = '/vols/opig/users/vavourakis/data/new_OAS_models/structures'
train_pdb_files = glob.glob(os.path.join(train_dir, '*.pdb'))

# cluster datasets by cdrh3 length
print('Clustering by CDRH3 length...')
train_metadata = pd.read_csv('/vols/opig/users/vavourakis/data/new_OAS_models/OAS_paired_filtered_newclust.csv')
train_metadata['cdr3_h_len'] = train_metadata.cdr3_h.apply(len)
train_cdr_cluster_ids = train_metadata.groupby('cdr3_h_len')['pdb_path'].apply(list).to_dict()

antibodies = parse_antibodies(pdb_files, n_jobs=-1)
cdr_clusters, cdr_cluster_ids = cluster_antibodies_by_CDR_length(antibodies, pdb_files, selection=[reg_def['CDRH3']])
struc_to_cdrh3_len = {pdb_path: cdr_length for cdr_length, pdb_paths in cdr_cluster_ids.items() for pdb_path in pdb_paths}

# get TM score gen_struc->{all train_strucs with same cdrh3 length}
job_pdb_file = sorted(pdb_files)[args.jobindex]
gen_cdrh3_len = struc_to_cdrh3_len[job_pdb_file]
train_strucs = train_cdr_cluster_ids[int(gen_cdrh3_len)]

max_tm_score, closest_train_struc = 0, None
for train_struc in tqdm(train_strucs):
    tm_score = get_TMscore(job_pdb_file, train_struc)
    if tm_score > max_tm_score and tm_score < 1:
        max_tm_score = tm_score
        closest_train_struc = train_struc     
row = [{'gen_struc': job_pdb_file, 'closest_train_struc': closest_train_struc, 'tm_score': max_tm_score}]
df = pd.DataFrame(row)
fname = f'closest_tm_score_partners_in_trainset_{args.jobindex}.csv'
df.to_csv(os.path.join(gen_dir, fname), index=False)

print('Done.')