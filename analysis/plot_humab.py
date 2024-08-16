# TODO: fix code duplication with number_ab_seqs.py

"""
Plots Hu-mAb results.

Example Usage:
    python plot_humab.py --gen_dir /vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun
"""

import pandas as pd
import numpy as np
import os, argparse
import matplotlib.pyplot as plt
import seaborn as sns

# from tqdm import tqdm

parser = argparse.ArgumentParser(description='Do species annotation on generated sequences with humab.')
parser.add_argument('--gen_dir', type=str, help='Path to directory with generated structures and sequences.', required=True)
args = parser.parse_args()
gen_dir = args.gen_dir

# load data
df = pd.read_csv(os.path.join(gen_dir, "humab_annotation.csv"))
df_trainorig = pd.read_csv("/vols/opig/users/vavourakis/generations/TRAINSET_origseq/humab_annotation.csv")
df_traingen = pd.read_csv("/vols/opig/users/vavourakis/generations/TRAINSET_genseq/humab_annotation.csv")

# plot
fig, axes = plt.subplots(nrows=1, ncols=4, dpi=300, figsize=(20, 5))
fig.suptitle('Percent Labelled "Human" by Hu-mAb', fontsize=24, y=1.00)  # Adjusted y to leave more space
chain_labels = ['Heavy Chain', 'Kappa Light Chain', 'Lambda Light Chain', 'Full Fv']

# to_consider = [df, df_trainorig, df_traingen]
to_consider = [df, df_trainorig]
for i, (title, frames) in enumerate([('VH Stats:', [f[f['chain'] == 'H'].copy() for f in to_consider]), 
                                    ('Kappa VL Stats:', [f[(f['chain'] == 'L') & (f['is_kappa'] == True)].copy() for f in to_consider]),
                                    ('Lambda VL Stats:', [f[(f['chain'] == 'L') & (f['is_kappa'] == False)].copy() for f in to_consider]),
                                    ('Full Fv Stats:', [f.copy() for f in to_consider])]): 
    
    # frame, frame_trainorig, frame_traingen = frames
    frame, frame_trainorig = frames

    # global stats
    pc_human = frame['human'].sum() / len(frame) * 100
    pc_human_trainorig = frame_trainorig['human'].sum() / len(frame_trainorig) * 100
    # pc_human_traingen = frame_traingen['human'].sum() / len(frame_traingen) * 100
    print('\n'+title)
    print(f'% human (de novo):\t {pc_human:.2f}%')
    print(f'% human (trainset):\t {pc_human_trainorig:.2f}%')
    # print(f'% human (traingen):\t {pc_human_traingen:.2f}%')

    # bin sequence lengths
    key = "len_tot" if title == 'Full Fv Stats:' else "seq_len"
    bins = np.linspace(min(frame[key]), max(frame[key])+1, 6)
    labels = [f'{int(bins[j])}-{int(bins[j+1])}' for j in range(len(bins)-1)]
    for f in frames:
        f['seq_len_bins'] = pd.cut(f[key], bins=bins, labels=labels, right=False)

    # stats per sequence-length bin
    pc_human_bins = frame.groupby('seq_len_bins')['human'].apply(lambda x: x.sum() / len(x) * 100)
    pc_human_trainorig_bins = frame_trainorig.groupby('seq_len_bins')['human'].apply(lambda x: x.sum() / len(x) * 100)
    # pc_human_traingen_bins = frame_traingen.groupby('seq_len_bins')['human'].apply(lambda x: x.sum() / len(x) * 100)

    # plot
    width = 0.2
    color = 'b' if i == 0 else 'r' if i in [1,2] else 'g'
    
    axes[i].set_title(chain_labels[i], fontsize=18)
    axes[i].set_ylim(0, 110)

    if i == 0:
        axes[i].set_ylabel('Percent', fontsize=14)  # Add y-axis label to the first subplot

    bars1 = axes[i].bar(range(len(pc_human_bins)), pc_human_bins.values, color=color, alpha=1, label='Generated AbMPNN', width=width)
    # bars2 = axes[i].bar([x + 2*width for x in range(len(pc_human_traingen_bins))], pc_human_traingen_bins.values, color='black', alpha=0.5, label='Trainset AbMPNN', width=width)
    bars3 = axes[i].bar([x + width for x in range(len(pc_human_trainorig_bins))], pc_human_trainorig_bins.values, color='black', alpha=1, label='Trainset Original', width=width)

    # overall bar
    overall_x = len(labels)  # position for the 'Overall' bar
    bars4 = axes[i].bar(overall_x, pc_human, color=color, alpha=1, width=width)
    # bars5 = axes[i].bar(overall_x + 2*width, pc_human_traingen, color='black', alpha=0.5, label='Trainset AbMPNN', width=width)
    bars6 = axes[i].bar(overall_x + width, pc_human_trainorig, color='black', alpha=1, width=width)
    # grey dotted line
    axes[i].axvline(x=overall_x - 0.5, color='grey', linestyle='dotted')
    # x-tick label for overall bar
    labels.append('Total')
    axes[i].set_xticks(range(len(labels)))
    axes[i].set_xticklabels(labels, ha='center', fontsize=14, rotation=35)
    
    # Add legend to each subplot
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(gen_dir, "designed_seqs", "humab_analysis.png"))