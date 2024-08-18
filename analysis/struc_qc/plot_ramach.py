"""
Plots Ramachandran densities for the generated Fv models
compared to the input dataset (train, val, and test combined).
Plots densities for VH, VL and whole Fv.

Relies on pre-computed phi-psi values for the reference set stored in 
vh_angles_train_val_test.csv and vl_angles_train_val_test.csv.

Usage:
python plot_ramach.py --gen_dir /vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun
"""

import os, warnings, argparse
import pandas as pd
from tqdm import tqdm

import Bio.PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from analysis.utils import overlay_ramachandran, sample_equivalent_trainset_strucs
from analysis.metrics import calculate_phi_psi_angles

parser = argparse.ArgumentParser(description='Process directories for Ramachandran plot generation.')
parser.add_argument('--gen_dir', type=str, default=None, help='Directory containing set of model generations to evaluate.')
parser.add_argument('--ref_dir', type=str, default="/vols/opig/users/vavourakis/data/ab_processed_newclust_newindex", help='Directory containing metadata file for reference set.')
args = parser.parse_args()

assert args.gen_dir is not None, "The --gen_dir argument must be specified and cannot be None."
gen_dir = args.gen_dir
ref_dir = args.ref_dir

out_dir = gen_dir

print("Finding sample data...")
gen_file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(gen_dir) for file in files if file == "sample.pdb"]

print("Loading reference data...")
train_file_paths = sample_equivalent_trainset_strucs(gen_file_paths, os.path.join(ref_dir, 'metadata.csv'))

print("Calculating phi-psi angles...")
chain_1_angles_all_models, chain_2_angles_all_models, vh_ref_angles, vl_ref_angles = [], [], [], []
for file_paths, chain_1_angles_list, chain_2_angles_list in [
    (gen_file_paths, chain_1_angles_all_models, chain_2_angles_all_models),
    (train_file_paths, vh_ref_angles, vl_ref_angles)
]:
    for pdb_file in tqdm(file_paths):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            structure = Bio.PDB.PDBParser().get_structure('structure', pdb_file)
            chain_1_angles, chain_2_angles = calculate_phi_psi_angles(structure)
            chain_1_angles_list.extend(chain_1_angles)
            chain_2_angles_list.extend(chain_2_angles)

print("Plotting Ramachandran Densities...")
print("... VH")
overlay_ramachandran(chain_1_angles_all_models, vh_ref_angles, 
                     title='VH Ramachandran Densities', 
                     color='b', labels=['Generated', 'Trainset'],
                     fname=os.path.join(out_dir, 'ramachandran_VH.png'))
print("... VL")
overlay_ramachandran(chain_2_angles_all_models, vl_ref_angles,
                    title='VL Ramachandran Densities',
                    color='r', labels=['Generated', 'Trainset'],
                    fname=os.path.join(out_dir, 'ramachandran_VL.png'))
print("... Fv")
overlay_ramachandran(chain_1_angles_all_models + chain_2_angles_all_models,
                     vh_ref_angles + vl_ref_angles,
                    title='Fv Ramachandran Densities',
                    color='g', labels=['Generated', 'Trainset'],
                    fname=os.path.join(out_dir, 'ramachandran_Fv.png'))