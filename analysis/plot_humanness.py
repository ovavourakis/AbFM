"""
Plot humanness score densities for generated sequences and trainset sequences.

Usage example:
    python plot_humanness.py --gen_dir /vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun --oasis_identity [--rerun_frame_construction]

Arguments:
    --gen_dir: Directory for generation files
"""

import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Process some directories.")
parser.add_argument('--gen_dir', type=str, help='Directory for generation files')
parser.add_argument('--rerun_frame_construction', action='store_true', help='Rerun the frame construction process')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--oasis_identity', action='store_true', help='Use OASis identity scoring')
group.add_argument('--oasis_percentile', action='store_true', help='Use OASis percentiles scoring')
args = parser.parse_args()
gen_dir = args.gen_dir

# globals
seq_dir = os.path.join(gen_dir, "designed_seqs")
gen_oasis_csv, traingen_oasis_csv, trainorig_oasis_csv = [os.path.join(seq_dir, f"oasis_humanness.csv"), 
    "/vols/opig/users/vavourakis/generations/TRAINSET_genseq/oasis_humanness.csv", 
    "/vols/opig/users/vavourakis/generations/TRAINSET_origseq/oasis_humanness.csv"]
anarci_csv, traingen_anarci_csv, trainorig_anarci_csv = [os.path.join(seq_dir, 'anarci_annotation.csv'), 
    "/vols/opig/users/vavourakis/generations/TRAINSET_genseq/anarci_annotation.csv", 
    "/vols/opig/users/vavourakis/generations/TRAINSET_origseq/anarci_annotation.csv"]

if args.rerun_frame_construction:
    # load data
    print("Loading data...")
    df_gen, df_traingen, df_trainorig = (pd.read_csv(f) for f in [gen_oasis_csv, traingen_oasis_csv, trainorig_oasis_csv])
    df_gen_anarci, df_traingen_anarci, df_trainorig_anarci = (pd.read_csv(f) for f in [anarci_csv, traingen_anarci_csv, trainorig_anarci_csv])

    # retrieve chain lengths from anarci outputs
    print("Getting chain lengths...")
    for df, df_anarci in tqdm(zip([df_gen, df_traingen, df_trainorig],
                                [df_gen_anarci, df_traingen_anarci, df_trainorig_anarci])):
        df[['structure', 'seqid']] = df.Antibody.apply(lambda x: pd.Series(x.rsplit('_', 1)))
        df[['len_h', 'len_l']] = pd.DataFrame(
            df.apply(
                lambda row: list(
                    df_anarci[
                        (df_anarci.structure == row['structure']) & 
                        (df_anarci.seq_id == int(row['seqid']) - 1)
                    ].seq_len
                ), 
                axis=1
            ).tolist(), 
            index=df.index
        )
        df['len_tot'] = df.len_h + df.len_l
        
        df_name = 'df_gen' if df is df_gen else 'df_traingen' if df is df_traingen else 'df_trainorig'
        df.to_csv(os.path.join(seq_dir, f'{df_name}.csv'), index=False)

        del df_anarci

# read the df_gen, df_traingen, df_trainorig from the saved CSVs
df_gen, df_traingen, df_trainorig = (pd.read_csv(os.path.join(seq_dir, f'{df_name}.csv')) for df_name in ['df_gen', 'df_traingen', 'df_trainorig'])

# stratify into buckets by chain lengths 
df_trainorig['len_tot_bucket'], bins_tot = pd.qcut(df_trainorig['len_tot'], 5, labels=False, retbins=True)
df_gen['len_tot_bucket'] = pd.cut(df_gen['len_tot'], bins_tot, labels=False, include_lowest=True)
df_traingen['len_tot_bucket'] = pd.cut(df_traingen['len_tot'], bins_tot, labels=False, include_lowest=True)

df_trainorig['len_h_bucket'], bins_h = pd.qcut(df_trainorig['len_h'], 5, labels=False, retbins=True)
df_gen['len_h_bucket'] = pd.cut(df_gen['len_h'], bins_h, labels=False, include_lowest=True)
df_traingen['len_h_bucket'] = pd.cut(df_traingen['len_h'], bins_h, labels=False, include_lowest=True)

df_trainorig['len_l_bucket'], bins_l = pd.qcut(df_trainorig['len_l'], 5, labels=False, retbins=True)
df_gen['len_l_bucket'] = pd.cut(df_gen['len_l'], bins_l, labels=False, include_lowest=True)
df_traingen['len_l_bucket'] = pd.cut(df_traingen['len_l'], bins_l, labels=False, include_lowest=True)

# what to plot
if args.oasis_identity:
    columns = ['OASis Identity', 'Heavy OASis Identity', 'Light OASis Identity']
elif args.oasis_percentile:
    columns = ['OASis Percentile', 'Heavy OASis Percentile', 'Light OASis Percentile']

# plot densities by chain lengths
fig, axes = plt.subplots(3, 5, figsize=(25, 15), sharey='row')
if args.oasis_identity:
    fig.suptitle('OASis 9-mer Identity (50% Prevalence Threshold) by Sequence Length', fontsize=24)
elif args.oasis_percentile:
    fig.suptitle('OASis Identity Therapeutic mAb Percentile (50% Prevalence Threshold) by Sequence Length', fontsize=24)
colors = [('green', 'black'), ('blue', 'black'), ('red', 'black')]
buckets = ['len_tot_bucket', 'len_h_bucket', 'len_l_bucket']
bins = [bins_tot, bins_h, bins_l]
labels = ['Generated AbMPNN', 'Trainset Original', 'Trainset AbMPNN']
labels_y = ['Full Fv', 'VH only', 'VL only']

for row, (column, (color1, color2), bucket, bin_edges) in enumerate(zip(columns, colors, buckets, bins)):
    for i in range(5):
        bucket_df_gen = df_gen[df_gen[bucket] == i]
        bucket_df_trainorig = df_trainorig[df_trainorig[bucket] == i]
        bucket_df_traingen = df_traingen[df_traingen[bucket] == i]
        sns.kdeplot(bucket_df_gen[column], ax=axes[row, i], color=color2, label=labels[0], clip=(0, 1), common_norm=True)
        sns.kdeplot(bucket_df_trainorig[column], ax=axes[row, i], color=color1, label=labels[1], alpha=0.5, clip=(0, 1), common_norm=True, fill=True)
        sns.kdeplot(bucket_df_traingen[column], ax=axes[row, i], color=color1, linestyle='--', label=labels[2], alpha=1, clip=(0, 1), common_norm=True)
        axes[row, i].set_xlim(0, 1)
        axes[row, i].set_title(f'Length {int(bin_edges[i])} - {int(bin_edges[i+1])}')
        if i == 0:
            axes[row, i].set_ylabel(labels_y[row], fontsize=24)
        axes[row, i].legend(loc='upper left')
        if args.oasis_identity:
            axes[row, i].set_xlabel('OASis 9-mer Identity (50% Prevalence Threshold)')
            axes[row, i].axvline(0.5, color='gray', linestyle='-')
            axes[row, i].text(0.5, 0.85, 'likely human', rotation=90, verticalalignment='center', horizontalalignment='right', color='gray', transform=axes[row, i].get_xaxis_transform())
            axes[row, i].axvline(0.8, color='gray', linestyle='-')
            axes[row, i].text(0.8, 0.81, 'very likely human', rotation=90, verticalalignment='center', horizontalalignment='right', color='gray', transform=axes[row, i].get_xaxis_transform())
    
plt.tight_layout(rect=[0, 0, 1, 0.95])
if args.oasis_identity:
    fig.savefig(os.path.join(seq_dir, 'oid_by_length.png'))
elif args.oasis_percentile:
    fig.savefig(os.path.join(seq_dir, 'opc_by_length.png'))

# plot overall densities 
fig_all, axes_all = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=150)
if args.oasis_identity:
    fig_all.suptitle('OASis 9-mer Identity (50% Prevalence Threshold)', fontsize=24)
elif args.oasis_percentile:
    fig_all.suptitle('OASis Identity Therapeutic mAb Percentile (50% Prevalence Threshold)', fontsize=24)

titles = ['Full Fv', 'VH only', 'VL only']
for row, (column, (color1, color2)) in enumerate(zip(columns, colors)):
    sns.kdeplot(df_gen[column], ax=axes_all[row], color=color2, label=labels[0], clip=(0, 1), common_norm=True)
    sns.kdeplot(df_trainorig[column], ax=axes_all[row], color=color1, label=labels[1], alpha=0.5, clip=(0, 1), common_norm=True, fill=True)
    sns.kdeplot(df_traingen[column], ax=axes_all[row], color=color1, linestyle='--', label=labels[2], alpha=1, clip=(0, 1), common_norm=True)
    axes_all[row].set_xlim(0, 1)
    axes_all[row].set_title(titles[row])
    axes_all[row].set_ylabel('Density')
    axes_all[row].legend(loc='upper left')
    if args.oasis_identity:
        axes_all[row].set_xlabel('OASis 9-mer Identity (50% Prevalence Threshold)')
        axes_all[row].axvline(0.5, color='gray', linestyle='-')
        axes_all[row].text(0.5, 0.85, 'likely human', rotation=90, verticalalignment='center', horizontalalignment='right', color='gray', transform=axes_all[row].get_xaxis_transform())
        axes_all[row].axvline(0.8, color='gray', linestyle='-')
        axes_all[row].text(0.8, 0.81, 'very likely human', rotation=90, verticalalignment='center', horizontalalignment='right', color='gray', transform=axes_all[row].get_xaxis_transform())

plt.tight_layout()
if args.oasis_identity:
    fig_all.savefig(os.path.join(seq_dir, 'oid.png'))
elif args.oasis_percentile:
    fig_all.savefig(os.path.join(seq_dir, 'opc.png'))