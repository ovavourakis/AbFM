""" Plots Hu-mAb results.

Example Usage:
    python plot_humab.py --gen_dir /vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun --train_orig /vols/opig/users/vavourakis/generations/TRAINSET_origseq2 --train_gen /vols/opig/users/vavourakis/generations/TRAINSET_genseq2
"""

import pandas as pd
import numpy as np
import os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')  # Use the 'Agg' backend for rendering LaTeX

parser = argparse.ArgumentParser(description='Do species annotation on generated sequences with humab.')
parser.add_argument('--gen_dir', type=str, help='Path to directory with generated structures and sequences.', required=True)
parser.add_argument('--train_orig', type=str, help='Path to directory with training structures and sequences.', required=True)
parser.add_argument('--train_gen', type=str, help='Path to directory with training structures and sequences.', required=True)
args = parser.parse_args()
gen_dir = args.gen_dir

# load data
df = pd.read_csv(os.path.join(gen_dir, "humab_annotation.csv"))
df_trainorig = pd.read_csv(os.path.join(args.train_orig, "humab_annotation.csv"))
df_traingen = pd.read_csv(os.path.join(args.train_gen, "humab_annotation.csv"))

# plot
fig, axes = plt.subplots(nrows=2, ncols=2, dpi=300, figsize=(15, 8))
# fig.suptitle('Percent Labelled "Human" by Hu-mAb', fontsize=24)  # Adjusted y to leave more space
chain_labels = ['Full Variable Region', 'Heavy Chain', r'$\kappa$ Light Chain', r'$\lambda$ Light Chain']

to_consider = [df, df_trainorig, df_traingen]
for i, (title, frames) in enumerate([('Full Fv Stats:', [f.copy() for f in to_consider]),
                                    ('VH Stats:', [f[f['chain'] == 'H'].copy() for f in to_consider]), 
                                    (r'$\kappa$ VL Stats:', [f[(f['chain'] == 'L') & (f['is_kappa'] == True)].copy() for f in to_consider]),
                                    (r'$\lambda$ VL Stats:', [f[(f['chain'] == 'L') & (f['is_kappa'] == False)].copy() for f in to_consider])]): 
    
    frame, frame_trainorig, frame_traingen = frames

    # global stats
    pc_human = frame['human'].sum() / len(frame) * 100
    pc_human_traingen = frame_traingen['human'].sum() / len(frame_traingen) * 100
    pc_human_trainorig = frame_trainorig['human'].sum() / len(frame_trainorig) * 100
    print('\n'+title)
    print(f'% human (de novo):\t {pc_human:.2f}%')
    print(f'% human (traingen):\t {pc_human_traingen:.2f}%')
    print(f'% human (trainset):\t {pc_human_trainorig:.2f}%')
    # print('\nN gensamples (de novo):\t', len(frame_trainorig))
    

    # bin sequence lengths
    key = "len_tot" if title == 'Full Fv Stats:' else "seq_len"
    bins = np.linspace(min(frame[key]), max(frame[key])+1, 6)
    labels = [f'{int(bins[j])}-{int(bins[j+1])}' for j in range(len(bins)-1)]
    for f in frames:
        f['seq_len_bins'] = pd.cut(f[key], bins=bins, labels=labels, right=False)

    # stats per sequence-length bin
    pc_human_bins = frame.groupby('seq_len_bins')['human'].apply(lambda x: x.sum() / len(x) * 100)
    pc_human_trainorig_bins = frame_trainorig.groupby('seq_len_bins')['human'].apply(lambda x: x.sum() / len(x) * 100)
    pc_human_traingen_bins = frame_traingen.groupby('seq_len_bins')['human'].apply(lambda x: x.sum() / len(x) * 100)

    # num_per_bin
    num_per_bin = frame['seq_len_bins'].value_counts()
    num_per_bin_trainorig = frame_trainorig['seq_len_bins'].value_counts()
    num_per_bin_traingen = frame_traingen['seq_len_bins'].value_counts()

    # plot
    width = 0.2
    color = 'g' if i == 0 else 'b' if i == 1 else 'r'
    
    ax = axes[i // 2, i % 2]
    ax.set_title(chain_labels[i], fontsize=18)
    ax.set_ylim(0, 110)

    if i == 0 or i == 2:
        ax.set_ylabel('Labelled Human (%)', fontsize=14)  # Add y-axis label to the first subplot in each row

    bars1 = ax.bar(range(len(pc_human_bins)), pc_human_bins.values, color=color, alpha=1, label='Generated AbMPNN', width=width)
    bars2 = ax.bar([x + width for x in range(len(pc_human_traingen_bins))], pc_human_traingen_bins.values, color='black', alpha=0.5, label='Trainset AbMPNN', width=width)
    bars3 = ax.bar([x + 2*width for x in range(len(pc_human_trainorig_bins))], pc_human_trainorig_bins.values, color='black', alpha=1, label='Trainset Original', width=width)

    # Add horizontal textbox labels with the number of observations, rotated by 90 degrees
    for bar, count in zip(bars1, num_per_bin):
        ax.text(bar.get_x() + bar.get_width() / 2, 3, f'{count}', ha='center', va='bottom', fontsize=10, rotation=90, color='white')
    for bar, count in zip(bars2, num_per_bin_traingen):
        ax.text(bar.get_x() + bar.get_width() / 2, 3, f'{count}', ha='center', va='bottom', fontsize=10, rotation=90, color='white')
    for bar, count in zip(bars3, num_per_bin_trainorig):
        ax.text(bar.get_x() + bar.get_width() / 2, 3, f'{count}', ha='center', va='bottom', fontsize=10, rotation=90, color='white')

    # overall bar
    overall_x = len(labels)  # position for the 'Overall' bar
    bars4 = ax.bar(overall_x, pc_human, color=color, alpha=1, width=width)
    bars5 = ax.bar(overall_x + width, pc_human_traingen, color='black', alpha=0.5, width=width)
    bars6 = ax.bar(overall_x + 2*width, pc_human_trainorig, color='black', alpha=1, width=width)
    
    # Add horizontal textbox labels to the overall bar
    ax.text(overall_x, 3, f'{len(frame)}', ha='center', va='bottom', fontsize=10, rotation=90, color='white')
    ax.text(overall_x + width, 3, f'{len(frame_traingen)}', ha='center', va='bottom', fontsize=10, rotation=90, color='white')
    ax.text(overall_x + 2*width, 3, f'{len(frame_trainorig)}', ha='center', va='bottom', fontsize=10, rotation=90, color='white')
    
    # grey dotted line
    ax.axvline(x=overall_x - 0.3, color='grey', linestyle='dotted')
    # x-tick label for overall bar
    labels.append('All')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, ha='center', fontsize=14)#, rotation=35)
    
    # Add legend to each subplot except the last one
    if i != len(chain_labels) - 1:
        ax.legend()

    # Add x-axis label to the lower two subplots
    if i == 2 or i == 3:
        ax.set_xlabel('Length', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(gen_dir, "designed_seqs", "humab_analysis.png"))