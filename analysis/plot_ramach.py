"""
Plots Ramachandran densities for the generated Fv models
compared to the input dataset (train, val, and test combined).
Plots densities for VH, VL and whole Fv.

Relies on pre-computed phi-psi values for the reference set stored in 
vh_angles_train_val_test.csv and vl_angles_train_val_test.csv.

Usage:
python plot_ramach.py --gen_dir /vols/opig/users/vavourakis/generations/NEWCLUST_newsample
"""

import os, warnings, argparse
import pandas as pd
from tqdm.auto import tqdm

import Bio.PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from analysis.utils import overlay_ramachandran
from analysis.metrics import calculate_phi_psi_angles

parser = argparse.ArgumentParser(description='Process directories for Ramachandran plot generation.')
parser.add_argument('--gen_dir', type=str, default=None, help='Directory containing set of model generations to evaluate.')
parser.add_argument('--ref_dir', type=str, default="/vols/opig/users/vavourakis/data/new_ab_processed", help='Directory containing reference angle CSV files (vh_angles_train_val_test.csv and vl_angles_train_val_test.csv).')
args = parser.parse_args()

assert args.gen_dir is not None, "The --gen_dir argument must be specified and cannot be None."
gen_dir = args.gen_dir
ref_dir = args.ref_dir

out_dir = gen_dir

print("Loading reference data...")
vh_ref_angles = pd.read_csv(os.path.join(ref_dir, "vh_angles_train_val_test.csv")).sample(frac=0.005, random_state=42).apply(tuple, axis=1).tolist()
vl_ref_angles = pd.read_csv(os.path.join(ref_dir, "vl_angles_train_val_test.csv")).sample(frac=0.005, random_state=42).apply(tuple, axis=1).tolist()

print("Loading sample data...")
all_file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(gen_dir) for file in files if file == "sample.pdb"]

chain_1_angles_all_models = []
chain_2_angles_all_models = []
for pdb_file in tqdm(all_file_paths):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        structure = Bio.PDB.PDBParser().get_structure('structure', pdb_file)
        chain_1_angles, chain_2_angles = calculate_phi_psi_angles(structure)
        chain_1_angles_all_models.extend(chain_1_angles)
        chain_2_angles_all_models.extend(chain_2_angles)

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