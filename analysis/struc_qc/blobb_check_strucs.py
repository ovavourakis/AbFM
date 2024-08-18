"""
Uses blobb_structure_check 
(https://biobb-structure-checking.readthedocs.io/en/latest/command_line_usage.html)
to evaluate a set of structures for common problems such as 
missing oxygens, backbone breaks, and backbone crosslinks.

Prints some synoptic statistics and outouts a plot of the 
distribution of these problems across different total lengths
in the dataset.

Usage: 
python blobb_check_strucs.py --num_processes 8 --pdb_dir /vols/opig/users/vavourakis/generations/NEWCLUST_last --rerun_check
"""

import os, argparse
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

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

print(pdb_dir)
print('\n')

# check the structures for chain breaks
with mp.Pool(processes=num_processes) as pool:
    results = pool.starmap(blobb_check, [(path, rerun_check) for path in all_file_paths])
    missing_o, bb_breaks, bb_bad_links, covalent_link_problems, total_residues = zip(*results)

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
    print(f"% with {metric_name}: \t\t {percentage:.2f}%")

# mean and std of problem counts per structure
print("\n")
for metric_name, metric_values in [("missing_o", missing_o), ("bb_breaks", bb_breaks)]:
    metric_values_non_zero = [value for value in metric_values if value > 0]
    metric_mean = sum(metric_values_non_zero) / len(metric_values_non_zero) if metric_values_non_zero else 0
    metric_std = (sum((i - metric_mean) ** 2 for i in metric_values) / len(metric_values)) ** 0.5
    print(f"Distribution #{metric_name}: \t\t {metric_mean:.2f} +/- {metric_std:.2f}")

# distributions across sequence length
df = pd.DataFrame({
    "missing_o": missing_o,
    "bb_breaks": bb_breaks,
    "bb_bad_links": bb_bad_links,
    "covalent_link_problems": covalent_link_problems,
    "total_residues": total_residues
})
df["bb_bad_links"] = df["bb_bad_links"].astype(int)

bins = [min(df["total_residues"])-1, 220, 230, 240, 250, max(df["total_residues"])+1]
labels = ["<220", "220-230", "230-240", "240-250", ">250"]
df['residue_bins'] = pd.cut(df['total_residues'], bins=bins, labels=labels, right=False)

bb_crosslinks_percentage = df.groupby('residue_bins')['bb_bad_links'].mean() * 100
mean_bb_breaks = df.groupby('residue_bins')['bb_breaks'].mean()
bb_breaks_percentage = df.groupby('residue_bins')['bb_breaks'].apply(lambda x: (sum(x > 0) / len(x)) * 100)

# CLI output
print("\npercent structures with backbone crosslinks by total length:")
print(bb_crosslinks_percentage)
print("\n percent structures with backbone breaks by total length:")
print(bb_breaks_percentage)
print("\nnumber of backbone breaks by total length:")
print(mean_bb_breaks)

# plot
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# fig.suptitle('Synoptic Structure QC', fontsize=16)

ax[0].set_title("Steric Clashes", fontsize=14)
bb_crosslinks_percentage.plot(kind='bar', ax=ax[0], color='blue', alpha=0.7)
ax[0].bar_label(ax[0].containers[0], fmt='%.0f', fontsize=12)
ax[0].set_ylim(0, 100)
ax[0].set_ylabel("prevalence (% of structures)", fontsize=12)

ax[1].set_title("Chain Breaks", fontsize=14)
bb_breaks_percentage.plot(kind='bar', ax=ax[1], color='green', alpha=0.7)
ax[1].bar_label(ax[1].containers[0], fmt='%.0f', fontsize=12)
ax[1].set_ylim(0, 100)
ax[1].set_ylabel("prevalence (% of structures)", fontsize=12)

ax[2].set_title("Chain Breaks per Structure", fontsize=14)
mean_bb_breaks.plot(kind='bar', ax=ax[2], color='red', alpha=0.7)
ax[2].bar_label(ax[2].containers[0], fmt='%.2f', fontsize=12)
ax[2].set_ylabel("number of breaks", fontsize=12)
max_value = mean_bb_breaks.max()
ax[2].set_ylim(0, max_value * 1.1)

for i in range(3):
    ax[i].set_xlabel("total sequence length (AA)", fontsize=12)
for axis in ax:
    axis.tick_params(axis='x', rotation=0, labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "synoptic_qc.png"),
            dpi=300)