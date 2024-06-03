"""
Uses blobb_structure_check 
(https://biobb-structure-checking.readthedocs.io/en/latest/command_line_usage.html)
to evaluate a set of structures for common problems such as 
missing oxygens, backbone breaks, and backbone crosslinks.

Prints some synoptic statistics and outouts a plot of the 
distribution of these problems across different total lengths
in the dataset.

Usage: 
python blobb_check_by_chain.py --num_processes 8 --pdb_dir /vols/opig/users/vavourakis/generations/NEWCLUST_last --rerun_check
"""

import os, argparse
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import tempfile
import shutil
from Bio.PDB import PDBParser, PDBIO

from metrics import blobb_check

parser = argparse.ArgumentParser(description='Process PDB files with blobb_structure_check.')
parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use.')
parser.add_argument('--pdb_dir', type=str, default=None, help='Directory containing generated structures.')
parser.add_argument('--rerun_check', action='store_true', help='Flag to rerun checks on PDB files (somewhat time-consuming).')

args = parser.parse_args()
num_processes = args.num_processes
pdb_dir = args.pdb_dir
if pdb_dir is None:
    raise ValueError("The --pdb_dir argument must be specified and cannot be None.")
rerun_check = args.rerun_check

out_dir = pdb_dir

all_file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(pdb_dir) for file in files if file == "sample.pdb"]

# Create a temporary directory to store the split pdb files
temp_dir = tempfile.mkdtemp()

# Split the pdb files into separate chains and save them to the temporary directory
for idx, pdb_file in enumerate(all_file_paths):
    parser = PDBParser()
    structure = parser.get_structure("pdb", pdb_file)
    for i, chain in enumerate(structure.get_chains()):
        io = PDBIO()
        io.set_structure(chain)
        io.save(os.path.join(temp_dir, f"sample_{idx}_chain_{i}.pdb"))

# Get the paths of the split pdb files for chain 1
all_file_paths_chain_1 = [os.path.join(root, file) for root, dirs, files in os.walk(temp_dir) for file in files if file.endswith("_chain_0.pdb")]

# check the structures for chain breaks in chain 1
with mp.Pool(processes=num_processes) as pool:
    results_chain_1 = pool.starmap(blobb_check, [(path, rerun_check) for path in all_file_paths_chain_1])
    missing_o_chain_1, bb_breaks_chain_1, bb_bad_links_chain_1, covalent_link_problems_chain_1, total_residues_chain_1 = zip(*results_chain_1)

# Get the paths of the split pdb files for chain 2
all_file_paths_chain_2 = [os.path.join(root, file) for root, dirs, files in os.walk(temp_dir) for file in files if file.endswith("_chain_1.pdb")]

# check the structures for chain breaks in chain 2
with mp.Pool(processes=num_processes) as pool:
    results_chain_2 = pool.starmap(blobb_check, [(path, rerun_check) for path in all_file_paths_chain_2])
    missing_o_chain_2, bb_breaks_chain_2, bb_bad_links_chain_2, covalent_link_problems_chain_2, total_residues_chain_2 = zip(*results_chain_2)

# check the structures for chain breaks in both chains combined
with mp.Pool(processes=num_processes) as pool:
    results_combined = pool.starmap(blobb_check, [(path, rerun_check) for path in all_file_paths])
    missing_o_combined, bb_breaks_combined, bb_bad_links_combined, covalent_link_problems_combined, total_residues_combined = zip(*results_combined)



fig, ax = plt.subplots(3, 3, figsize=(12, 9))

for chain_num, results in enumerate([(missing_o_combined, bb_breaks_combined, covalent_link_problems_combined, bb_bad_links_combined, total_residues_combined),
                                     (missing_o_chain_1, bb_breaks_chain_1, covalent_link_problems_chain_1, bb_bad_links_chain_1, total_residues_chain_1), 
                                     (missing_o_chain_2, bb_breaks_chain_2, covalent_link_problems_chain_2, bb_bad_links_chain_2, total_residues_chain_2)]):
    missing_o, bb_breaks, covalent_link_problems, bb_bad_links, total_residues = results
    metrics = [("missing oxygens", missing_o), 
               ("backbone breaks", bb_breaks), 
               ("covalent link problems", covalent_link_problems),
               ("backbone cross-links", bb_bad_links)
    ]

    # percentage of structures with problems
    for metric_name, metric_values in metrics:
        if metric_name in ["missing oxygens", "backbone breaks"]:       # These metrics are counts, need to check if > 0
            percentage = (sum(i > 0 for i in metric_values) / len(metric_values)) * 100
        else:                                                           # These metrics are boolean flags, directly sum them up
            percentage = (sum(metric_values) / len(metric_values)) * 100
        print(f"Chain {chain_num} % with {metric_name}: \t\t {percentage:.2f}%")

    # mean and std of problem counts per structure
    print("\n")
    for metric_name, metric_values in [("missing_o", missing_o), ("bb_breaks", bb_breaks)]:
        metric_values_non_zero = [value for value in metric_values if value > 0]
        metric_mean = sum(metric_values_non_zero) / len(metric_values_non_zero) if metric_values_non_zero else 0
        metric_std = (sum((i - metric_mean) ** 2 for i in metric_values) / len(metric_values)) ** 0.5
        print(f"Chain {chain_num} Distribution #{metric_name}: \t\t {metric_mean:.2f} +/- {metric_std:.2f}")

    # distributions across sequence length
    df = pd.DataFrame({
        "missing_o": missing_o,
        "bb_breaks": bb_breaks,
        "bb_bad_links": bb_bad_links,
        "covalent_link_problems": covalent_link_problems,
        "total_residues": total_residues
    })
    df["bb_bad_links"] = df["bb_bad_links"].astype(int)

    min_residues = min(df["total_residues"])
    max_residues = max(df["total_residues"])
    bins = pd.cut(df['total_residues'], bins=5, labels=False, retbins=True)[1]
    bins[0] = min_residues
    bins[-1] = max_residues
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    labels = [str(i) for i in range(len(labels))] if len(set(labels)) != len(labels) else labels
    df['residue_bins'] = pd.cut(df['total_residues'], bins=bins, labels=labels, right=False, ordered=False)

    bb_crosslinks_percentage = df.groupby('residue_bins')['bb_bad_links'].mean() * 100
    mean_bb_breaks = df.groupby('residue_bins')['bb_breaks'].mean()
    bb_breaks_percentage = df.groupby('residue_bins')['bb_breaks'].apply(lambda x: (sum(x > 0) / len(x)) * 100 if len(x) != 0 else 0)

    # CLI output
    print(f"\nChain {chain_num+1} percent structures with backbone crosslinks by total length:")
    print(bb_crosslinks_percentage)
    print(f"\nChain {chain_num+1} percent structures with backbone breaks by total length:")
    print(bb_breaks_percentage)
    print(f"\nChain {chain_num+1} number of backbone breaks by total length:")
    print(mean_bb_breaks)

    # plot
    colour = 'g' if chain_num == 0 else 'b' if chain_num == 1 else 'r'
    if chain_num < 1:
        ax[chain_num][0].set_title(f"Steric Clashes", fontsize=14)
    bb_crosslinks_percentage.plot(kind='bar', ax=ax[chain_num][0], color=colour, alpha=0.5)
    ax[chain_num][0].bar_label(ax[chain_num][0].containers[0], fmt='%.0f', fontsize=12)
    ax[chain_num][0].set_ylim(0, 110)
    ylab = 'Entire Fv' if chain_num == 0 else 'VH' if chain_num == 1 else 'VL'
    ax[chain_num][0].set_ylabel(ylab, fontsize=18)
    if chain_num == 2:
        ax[chain_num][0].set_xlabel("total sequence length (AA)", fontsize=12)

    if chain_num < 1:
        ax[chain_num][1].set_title(f"Chain Breaks", fontsize=14)
    bb_breaks_percentage.plot(kind='bar', ax=ax[chain_num][1], color=colour, alpha=0.5)
    ax[chain_num][1].bar_label(ax[chain_num][1].containers[0], fmt='%.0f', fontsize=12)
    ax[chain_num][1].set_ylim(0, 100)
    ax[chain_num][1].set_ylabel("prevalence (% of structures)", fontsize=12)
    if chain_num == 2:
        ax[chain_num][1].set_xlabel("total sequence length (AA)", fontsize=12)

    if chain_num < 1:
        ax[chain_num][2].set_title(f"Chain Breaks per Structure", fontsize=14)
    mean_bb_breaks.plot(kind='bar', ax=ax[chain_num][2], color=colour, alpha=0.5)
    ax[chain_num][2].bar_label(ax[chain_num][2].containers[0], fmt='%.2f', fontsize=12)
    ax[chain_num][2].set_ylabel("number of breaks", fontsize=12)
    ax[chain_num][2].set_ylim(0, 0.35)
    if chain_num == 2:
        ax[chain_num][2].set_xlabel("total sequence length (AA)", fontsize=12)
    for axis in ax[chain_num]:
        axis.tick_params(axis='x', rotation=0, labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "synoptic_qc_combined.png"), dpi=300)
