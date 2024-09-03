import os, argparse, ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from anarci import anarci
from Bio import SeqIO

def count_res_per_region(row, use_north=True):
    chain = row['chain']
    numbering = ast.literal_eval(row['domain_numbering'])

    count_fw1 = count_cdr1 = count_fw2 = count_cdr2 = count_fw3 = count_cdr3 = count_fw4 = 0
    for (pos, ins_code), aa in numbering:
        if use_north and chain == 'H':
            count_fw1 += int(pos <= 23 and aa != '-')
            count_cdr1 += int(pos > 23 and pos <= 40 and aa != '-')
            count_fw2 += int(pos > 40 and pos <= 54 and aa != '-')
            count_cdr2 += int(pos > 54 and pos <= 66 and aa != '-')
            count_fw3 += int(pos > 66 and pos <= 104 and aa != '-')
            count_cdr3 += int(pos > 104 and pos <= 117 and aa != '-')
            count_fw4 += int(pos > 117 and aa != '-') 
        elif use_north and chain == 'L':
            count_fw1 += int(pos <= 23 and aa != '-')
            count_cdr1 += int(pos > 23 and pos <= 40 and aa != '-')
            count_fw2 += int(pos > 40 and pos <= 54 and aa != '-')
            count_cdr2 += int(pos > 54 and pos <= 69 and aa != '-')
            count_fw3 += int(pos > 69 and pos <= 104 and aa != '-')
            count_cdr3 += int(pos > 104 and pos <= 117 and aa != '-')
            count_fw4 += int(pos > 117 and aa != '-') 
        else:
            count_fw1 += int(pos <= 26 and aa != '-')
            count_cdr1 += int(pos > 26 and pos <= 39 and aa != '-')
            count_fw2 += int(pos > 39 and pos <= 55 and aa != '-')
            count_cdr2 += int(pos > 55 and pos <= 65 and aa != '-')
            count_fw3 += int(pos > 65 and pos <= 104 and aa != '-')
            count_cdr3 += int(pos > 104 and pos <= 117 and aa != '-')
            count_fw4 += int(pos > 117 and aa != '-')
    return count_fw1, count_cdr1, count_fw2, count_cdr2, count_fw3, count_cdr3, count_fw4

gen_dir = '/vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun'
trainset_orig = '/vols/opig/users/vavourakis/generations/TRAINSET_origseq2'
trainset_gens = '/vols/opig/users/vavourakis/generations/TRAINSET_genseq2'

# df_gen = pd.read_csv(os.path.join(gen_dir, "designed_seqs", "anarci_annotation.csv"))
# df_train_orig = pd.read_csv(os.path.join(trainset_orig, "anarci_annotation.csv"))
# df_train_gens = pd.read_csv(os.path.join(trainset_gens, "anarci_annotation.csv"))

# for i, df in enumerate([df_gen, df_train_orig, df_train_gens]):
#     df = df.dropna(subset=['domain_numbering'])
#     region_counts = df.apply(count_res_per_region, axis=1).apply(pd.Series)
#     region_counts.columns = ['lenFR1', 'lenCDR1', 'lenFR2', 'lenCDR2', 'lenFR3', 'lenCDR3', 'lenFR4']
#     df = pd.concat([df, region_counts], axis=1)
#     if i == 0:
#         df_gen = df
#         df_gen.to_csv(os.path.join(gen_dir, "region_lengths.csv"), index=False)
#     elif i == 1:
#         df_train_orig = df
#         df_train_orig.to_csv(os.path.join(gen_dir, "region_lengths_trainorig.csv"), index=False)
#     else:
#         df_train_gens = df
#         df_train_gens.to_csv(os.path.join(gen_dir, "region_lengths_traingen.csv"), index=False)

df_gen = pd.read_csv(os.path.join(gen_dir, "region_lengths.csv"))
df_train_orig = pd.read_csv(os.path.join(gen_dir, "region_lengths_trainorig.csv"))
df_train_gens = pd.read_csv(os.path.join(gen_dir, "region_lengths_traingen.csv"))

for chain_type, chain_label in [('H', 'Heavy Chain'), ('L', 'Light Chain')]:
    
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    
    bar_width = 0.1
    colors = ['red' if chain_type == 'L' else 'blue', 'gray', 'black']
    for df, offset, color, label in zip([df_gen, df_train_gens, df_train_orig], np.arange(3) * bar_width, colors, ['Generated AbMPNN', 'Trainset AbMPNN', 'Trainset Original']):
        frame = df[df['chain'] == chain_type]
       
        domain_presence = frame[['lenFR1', 'lenCDR1', 'lenFR2', 'lenCDR2', 'lenFR3', 'lenCDR3', 'lenFR4']].gt(0).mean() * 100

        bars = axes[0, 0].bar(domain_presence.index, domain_presence.values, color=color, alpha=1, width=bar_width, label=label, edgecolor=None)
        for bar in bars:
            bar.set_x(bar.get_x() + offset)
            bar.set_width(bar_width)
            bar.set_color(color)
            bar.set_edgecolor('black')
            bar.set_alpha(1)
            bar.set_label(label)
        
        axes[0, 0].set_title('Domain Presence (%)', fontsize=16)
        axes[0, 0].set_ylabel('Structures Containing Domain (%)', fontsize=16)
        axes[0, 0].set_yticklabels([f'{int(y)}' for y in axes[0, 0].get_yticks()], fontsize=14)
        axes[0, 0].set_xticklabels(['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4'], ha='center', rotation=15, fontsize=14)
        max_value = domain_presence.max()
        axes[0, 0].set_ylim(0, max_value * 1.05)
        axes[0, 0].set_xlim(0-5*bar_width, 6+5*bar_width)

    bar_width = 0.25
    titles = ['CDR1', 'CDR2', 'CDR3', 'FR1', 'FR2', 'FR3', 'FR4']
    max_bins = max(df_gen['len'+title].nunique() for title in titles)
    x_axis_bounds = {   'CDR1': (5, 21),
                        'CDR2': (2, 18),
                        'CDR3': (3, 26),
                        'FR1': (19, 35),
                        'FR2': (8, 24),
                        'FR3': (28, 44),
                        'FR4': (0, 16)
                    }
    for i, title in enumerate(titles, start=1):
        row, col = divmod(i, 4)
        for df, offset, color, label in zip([df_gen, df_train_orig, df_train_gens], np.arange(3) * bar_width, colors, ['Generated AbMPNN', 'Trainset AbMPNN', 'Trainset Original ']):
            frame = df[df['chain'] == chain_type]
            length_counts = frame['len' + title].value_counts(normalize=True).sort_index() * 100
            bars = axes[row, col].bar(length_counts.index + offset, length_counts.values, color=color, alpha=1, width=bar_width, label=label, edgecolor='black')
        
        adjusted_x_axis_bounds = (x_axis_bounds[title][0] - 0.5, x_axis_bounds[title][1] + 0.5)
        axes[row, col].set_xlim(adjusted_x_axis_bounds)
        axes[row, col].set_title(f'{title} Length Distribution', fontsize=16)
        if row == 1:
            axes[row, col].set_xlabel('Sequence Length (AA)', fontsize=16)
        else:
            axes[row, col].set_xlabel('')
        if (col == 1 and row == 0) or (col == 0 and row == 1):
            axes[row, col].set_ylabel('Percentage (%)', fontsize=16)
        else:
            axes[row, col].set_ylabel('')
        max_value = 100 #max(length_counts.max() for df in [df_gen, df_train_orig, df_train_gens] for length_counts in [df[df['chain'] == chain_type]['len' + title].value_counts(normalize=True).sort_index() * 100])
        axes[row, col].set_ylim(0, max_value * 1.05)
        x_ticks = range(x_axis_bounds[title][0], x_axis_bounds[title][1] + 1)
        axes[row, col].set_xticks(x_ticks)
        axes[row, col].set_xticklabels(x_ticks, ha='center', fontsize=14, rotation=90)
        axes[row, col].set_yticklabels([ f'{int(y)}' for y in axes[0, 0].get_yticks()], fontsize=14)

    axes[0, 3].legend(loc='upper right')  # legend in top right subplot
    plt.tight_layout()
    plt.savefig(os.path.join(gen_dir, f"region_analysis_{chain_type}.png"))