# TODO: remove code duplication with data_module.py and analysis utils

"""
This script samples unique VH/VL-length combinations from a reference dataset of antibody sequences.
The reference dataset is filtered to include only sequences within specified length ranges for both
heavy and light chains. The script then reconstructs the middle 95% of the VH and VL length distributions
from the filtered dataset and removes extremal combinations. Finally, it samples a set number of unique
VH/VL-length combinations based on the reconstructed distributions, 
if the sampled combination exists in the reference set.

This mimics the sampling process during de novo generation and is used to derive comparator property distirbutions.

Globals:
    metadata_csv (str): Path to the CSV file containing the reference dataset.
    min_length (int): Minimum total length of the antibody sequence.
    max_length (int): Maximum total length of the antibody sequence.
    min_length_heavy (int): Minimum length of the heavy chain.
    max_length_heavy (int): Maximum length of the heavy chain.
    min_length_light (int): Minimum length of the light chain.
    max_length_light (int): Maximum length of the light chain.

Functions:
    get_valid_lens_probs(df, column='len_h'):
        Takes lengths of individual chain (VH or VL) in reference dataset.
        Returns lengths within the middle 95% of that distribution and their probabilities.

Usage:
    This script is intended to be run as a standalone module.
    
    Example CLI usage:
        python analysis/trainset_lencombo_sampler.py --metadata_csv /path/to/metadata.csv
"""

import pandas as pd
import numpy as np
import argparse
from analysis.utils import get_valid_lens_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample unique VH/VL-length combinations from a reference dataset of antibody sequences.')
    parser.add_argument('--metadata_csv', type=str, default='/vols/opig/users/vavourakis/data/ab_processed_newclust_newindex/metadata.csv', help='Path to the CSV file containing the reference dataset.')
    args = parser.parse_args()

    # globals
    metadata_csv = args.metadata_csv
    min_length, max_length = 215, 260
    min_length_heavy, max_length_heavy = 109, 145
    min_length_light, max_length_light = 99, 121
    num_combos = 100
    num_samples_per_combo = 20

    # read in reference dataset on which to base sampling distribution
    df = pd.read_csv(metadata_csv)
    seqs = df['full_seq'].str.split('/', expand=True)
    df['len_h'], df['len_l'] = seqs[0].str.len(), seqs[1].str.len()
    df['len_tot'] = df['len_h'] + df['len_l']
    # apply hard filters
    df = df[(df['len_tot'] <= max_length) & (df['len_tot'] >= min_length)]
    df = df[(df['len_h'] <= max_length_heavy) & (df['len_h'] >= min_length_heavy)]
    df = df[(df['len_l'] <= max_length_light) & (df['len_l'] >= min_length_light)]

    # reconstruct middle 95% of VH and VL length distros from reference dataset
    lengths_h, probabilities_h = get_valid_lens_probs(df, 'len_h')
    lengths_l, probabilities_l = get_valid_lens_probs(df, 'len_l')

    # remove extremal combinations from dataset
    df = df[df.len_h.apply(lambda l: l in lengths_h)]
    df = df[df.len_l.apply(lambda l: l in lengths_l)]

    # sample a set number of unique VH/VL-length combinations
    dataset_combos = list(df['full_seq'].apply(lambda s: tuple([len(x) for x in s.split('/')])))
    sampled_combos = set()
    while len(sampled_combos) < num_combos:
        h = np.random.choice(lengths_h, p=probabilities_h)
        l = np.random.choice(lengths_l, p=probabilities_l)
        if (
            h + l >= min_length and 
            h + l <= max_length and 
            (h, l) not in sampled_combos and 
            (h, l) in dataset_combos
        ):
            sampled_combos.add((h,l)) 

    # sample N examples for every combination
    sample_paths = []
    for combo in sampled_combos:
        dfc = df[(df.len_h == combo[0]) & (df.len_l == combo[1])]
        sampled_rows = dfc.sample(n=num_samples_per_combo, replace=False)
        sample_paths.extend(list(sampled_rows.raw_path))

    for path in sample_paths:
        print(path)