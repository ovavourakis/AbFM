import os
import multiprocessing as mp
import matplotlib.pyplot as plt

from metrics import blobb_check

num_processes = 8
pdb_dir = "/data/localhost/not-backed-up/nvme00/vavourakis/codebase/AbFM/pwcutoff_inference"
all_file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(pdb_dir) for file in files if file == "sample.pdb"]

print(pdb_dir)

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




import pandas as pd

# Convert results to DataFrame for easier manipulation
df = pd.DataFrame({
    "missing_o": missing_o,
    "bb_breaks": bb_breaks,
    "bb_bad_links": bb_bad_links,
    "covalent_link_problems": covalent_link_problems,
    "total_residues": total_residues
})

# Convert bb_bad_links to numeric for percentage calculation
df["bb_bad_links"] = df["bb_bad_links"].astype(int)

# Define bins for total_residues
bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, max(df["total_residues"])+1]
labels = ['0-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000', '1001+']
df['residue_bins'] = pd.cut(df['total_residues'], bins=bins, labels=labels, right=False)

# Calculate percentage of bb_bad_links in each bin
bb_bad_links_percentage = df.groupby('residue_bins')['bb_bad_links'].mean() * 100

print("\nPercentage of structures with unexpected backbone links by total residues:")
print(bb_bad_links_percentage)

# Plotting
plt.figure(figsize=(10, 5))

# Scatter plot for bb_breaks vs total_residues
plt.subplot(1, 2, 1)
plt.scatter(df["total_residues"], df["bb_breaks"], alpha=0.5)
plt.title("BB Breaks vs Total Residues")
plt.xlabel("Total Residues")
plt.ylabel("BB Breaks")

plt.tight_layout()
plt.savefig()

