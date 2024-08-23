""" Script to sub-sample the reference set (e.g. trainset) with a distribution of 
length combinations exactly equiavlent to a set of generated structures. 

Usage:
    python make_equiv_train_subsample.py --gen_dir /vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun --ref_dir /vols/opig/users/vavourakis/data/ab_processed_newclust_newindex
"""

import os, argparse
from analysis.utils import sample_equivalent_trainset_chainlen, sample_equivalent_trainset_lencdrh3

parser = argparse.ArgumentParser(description='Subsample an equivalent set of structures from trainset.')
parser.add_argument('--gen_dir', type=str, default=None, help='Directory containing set of model generations to evaluate.')
parser.add_argument('--ref_dir', type=str, default="/vols/opig/users/vavourakis/data/ab_processed_newclust_newindex", help='Directory containing metadata file for reference set.')
args = parser.parse_args()

assert args.gen_dir is not None, "The --gen_dir argument must be specified and cannot be None."
gen_dir = args.gen_dir
ref_dir = args.ref_dir

out_dir = gen_dir

# find sample data
gen_file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(gen_dir) for file in files if file == "sample.pdb"]

# load reference data
train_file_paths = sample_equivalent_trainset_lencdrh3(gen_dir, factor=1)
# train_file_paths = sample_equivalent_trainset_chainlen(gen_file_paths, os.path.join(ref_dir, 'metadata.csv'), factor=1)
# train_file_paths = sample_equivalent_trainset_chainlen(gen_file_paths, os.path.join(ref_dir, 'metadata.csv'), factor=10)

for path in train_file_paths:
    print(path)