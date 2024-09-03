"""
Calculates pairwise RMSDs among length-homogenous CDRs for each CDR type using IMGT-numbered PDB files.
Generate a violin plot to visualize the distribution of RMSD values for each CDR type.

Usage:
    python diversity.py
"""

import glob, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from SPACE2.util import reg_def
from SPACE2.exhaustive_clustering import get_distance_matrices

import matplotlib.pyplot as plt
import seaborn as sns

# define IMGT-numbered pdb files
gen_dir = '/vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun'
pdb_files = glob.glob(os.path.join(gen_dir, 'renumbered_pdbs', '*.pdb'))

# calculate pairwise RMSDs among length-homogenous CDRs for each CDR type
cdrs = ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']
rmsd_matrices = {cdr: {} for cdr in cdrs}
for cdr in tqdm(['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']):
    selection = [reg_def[cdr]]
    out = get_distance_matrices(pdb_files, selection=selection, anchors=selection, n_jobs=-1)
    rmsd_matrices[cdr] = [val[1] for key, val in out.items()] 

# convert to flat list of rmsd values per cdr
rmsd_values = {cdr: [value for matrix in rmsd_matrices[cdr] for value in matrix[np.triu_indices(len(matrix), k=1)]] for cdr in cdrs}

# plot
plt.figure(figsize=(7.5, 6), dpi=300)
sns.violinplot(data=[rmsd_values[cdr] for cdr in cdrs], palette="Set3", cut=0)
plt.xticks(ticks=range(len(cdrs)), labels=cdrs)
plt.ylabel('RMSD (Angstrom)')
plt.title('Pairwise Inter-Sample RMSDs for Same-Length North CDRs', fontsize=16)
outfile = os.path.join(gen_dir, 'designed_seqs', 'pw_rmsd_violinplot.png')
plt.tight_layout()
plt.savefig(outfile)

