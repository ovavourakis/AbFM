import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd

from metrics import blobb_check

num_processes = 8
pdb_dir = "/vols/opig/users/vavourakis/generations/ftnl_auxloss_lastquarter_inference"
all_file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(pdb_dir) for file in files if file == "sample.pdb"]

print(pdb_dir)
print('\n')

# check the structures for chain breaks
with mp.Pool(processes=num_processes) as pool:
    results = pool.map(blobb_check, all_file_paths)
    missing_o, bb_breaks, bb_bad_links, covalent_link_problems, total_residues = zip(*results)

metrics = [("missing oxygens", missing_o), 
           ("backbone breaks", bb_breaks), 
           ("covalent link problems", covalent_link_problems),
           ("backbone cross-links", bb_bad_links), 
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
    metric_mean = sum(metric_values) / len(metric_values)
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

print("\npercent structures with backbone crosslinks by total length:")
print(bb_crosslinks_percentage)

print("\n percent structures with backbone breaks by total length:")
print(bb_breaks_percentage)

print("\nnumber of backbone breaks by total length:")
print(mean_bb_breaks)

# plot
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Synoptic QC')  # Added title
bb_crosslinks_percentage.plot(kind='bar', ax=ax[0], color='blue', alpha=0.7)
ax[0].set_ylim(0, 70)
bb_breaks_percentage.plot(kind='bar', ax=ax[1], color='green', alpha=0.7)
ax[1].set_ylim(0, 70)
mean_bb_breaks.plot(kind='line', ax=ax[2], color='red', alpha=0.7)
ax[0].set_ylabel("% of structures with backbone crosslinks")
ax[1].set_ylabel("% of structures with backbone breaks")
ax[2].set_ylabel("mean #backbone breaks")
for axis in ax:
    axis.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig("distro.png")

