"""
Performs sanity-check quality control on antibody sequences generated from trainset structures, using ANARCI. 
We require that generated sequences be human-antibody-like, that is:
- numberable by ANARCI
- recognised as single-domain by ANARCI (one domain per chain)
- recognised as human by ANARCI
- correctly inferred as the intended chain type (H, K, or L)

The directory containing the generated sequences is passed as input and 
should be structured as follows:

- gen_dir/
    - seqs/
        - <structure_name>.fa
Usage:
    python trainset_number_ab_seqs.py --gen_dir <path_to_root_of_generated_structures_and_seqs> [--rerun_annotation]
    python trainset_number_ab_seqs.py --gen_dir /vols/opig/users/vavourakis/generations/TRAINSET_genseq2 --rerun_annotation

Arguments:
    --gen_dir: Path to the directory containing generated sequences.
    --rerun_annotation: Optional flag to rerun ANARCI annotation on sequences (use on first run).

Output:
    - Console output of QC metrics.
    - Plots summarizing the QC analysis.
"""

import os, argparse, ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from anarci import anarci
from Bio import SeqIO

def chain_types_match(type, intended):
    if (type == 'H' and intended == 'H' or 
        type == 'K' and intended == 'L' or 
        type == 'L' and intended == 'L'):
        return True
    return False

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

parser = argparse.ArgumentParser(description='Run overview QC on AbMPNN-sampled sequences for the trainset structures.')
parser.add_argument('--gen_dir', type=str, help='Path to directory with generated structures and sequences.', required=True)
parser.add_argument('--rerun_annotation', action='store_true', help='Flag to rerun ANARCI annotation on sequences (somewhat time-consuming).', default=False)
args = parser.parse_args()
gen_dir = args.gen_dir

# annotate using anarci -----------------------------------------------------------------------------------------------
if args.rerun_annotation:
    print('Annotating Sequences...')  
    df = pd.DataFrame()
    fasta_dir = os.path.join(gen_dir, "seqs")
    all_fastas = [f for f in os.listdir(fasta_dir) if f.endswith('.fa')]
    for f in tqdm(all_fastas):
        with open(os.path.join(fasta_dir, f)) as handle:
            struc_name = f.split(".")[0]

            seq_id = -1
            for record in SeqIO.parse(handle, "fasta"):
                seq_id += 1

                sequences = [ (f'H', str(record.seq).split("/")[0]), 
                            (f'L', str(record.seq).split("/")[1]) ]
                numbering, alignment_details, hit_tables = anarci(sequences, scheme="imgt", output=False)

                for seq in range(len(sequences)):
                    intended_chain_type = sequences[seq][0]
                    sqnce = sequences[seq][1]
                    seq_len = len(sequences[seq][1])
                    
                    numbering_failure = True if numbering[seq] is None else False
                    single_domain = False if numbering_failure or len(numbering[seq]) > 1 else True
                    
                    if single_domain:
                        for domain in range(len(numbering[seq])): # should have length 1
                            human = True if alignment_details[seq][domain]['species'] == 'human' else False
                            correct_type = chain_types_match(alignment_details[seq][domain]['chain_type'], 
                                                            intended_chain_type)
                            domain_numbering, start_index, end_index = numbering[seq][domain]
                    else:
                        human, correct_type = False, False
                        domain_numbering, start_index, end_index = None, None, None

                    new_row = {'structure': struc_name,
                            'seq_id': seq_id,
                            'chain': intended_chain_type,
                            'numbering_failure': numbering_failure,
                            'single_domain': single_domain,
                            'human': human,
                            'correct_type': correct_type,
                            'domain_numbering': domain_numbering,
                            'start_index': start_index,
                            'end_index': end_index,
                            'seq_len': seq_len,
                            'seq': sqnce
                            }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(os.path.join(gen_dir, "anarci_annotation.csv"), index=False)

# plot sequence qc -----------------------------------------------------------------------------------------------
df = pd.read_csv(os.path.join(gen_dir, "anarci_annotation.csv"))

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17, 9), dpi=300)
# fig.suptitle('Synoptic Sequence QC', fontsize=24, y=0.96)
titles = ['Percent Successfully \n Numbered', 'Percent Recognised as \n Single-Domain', 'Percent Recognised as \n Human', 'Percent with Matching \n Inferred Locus']
chain_labels = ['Heavy Chain', 'Light Chain']

for i, (title, frame) in enumerate([('Heavy-Chain Stats:', df[df['chain'] == 'H'].copy()), 
                                    ('Light-Chain Stats:', df[df['chain'] == 'L'].copy()),
                                    ('All-Chain Stats:', df)]):
    
    # calculate and print global stats
    pc_numbered = 100 - frame['numbering_failure'].sum() / len(frame) * 100
    pc_single_domain = frame['single_domain'].sum() / len(frame) * 100
    pc_human = frame['human'].sum() / len(frame) * 100
    pc_correct_type = frame['correct_type'].sum() / len(frame) * 100

    print('\n'+title)
    print(f'% numbered:\t\t {pc_numbered:.2f}%')
    print(f'% single domain:\t {pc_single_domain:.2f}%')
    print(f'% human:\t\t {pc_human:.2f}%')
    print(f'% correct chain type:\t {pc_correct_type:.2f}%')

    if i in [0, 1]:

        # bin sequence lengths
        bins = np.linspace(min(frame["seq_len"]), max(frame["seq_len"])+1, 6)
        # bins = np.linspace(min(frame["seq_len"]), max(frame["seq_len"])+1, 3)
        labels = [f'{int(bins[j])}-{int(bins[j+1])}' for j in range(len(bins)-1)]
        frame['seq_len_bins'] = pd.cut(frame['seq_len'], bins=bins, labels=labels, right=False)

        # stats per sequence-length bin
        pc_numbered_bins = frame.groupby('seq_len_bins')['numbering_failure'].apply(lambda x: 100 - x.sum() / len(x) * 100)
        pc_single_domain_bins = frame.groupby('seq_len_bins')['single_domain'].apply(lambda x: x.sum() / len(x) * 100)
        pc_human_bins = frame.groupby('seq_len_bins')['human'].apply(lambda x: x.sum() / len(x) * 100)
        pc_correct_type_bins = frame.groupby('seq_len_bins')['correct_type'].apply(lambda x: x.sum() / len(x) * 100)

        # plot by-length stats for this chain type
        metrics = [pc_numbered_bins, pc_single_domain_bins, pc_human_bins, pc_correct_type_bins]
        for j, metric in enumerate(metrics):
            bars = axes[i, j].bar(metric.index, metric.values, color='b' if i == 0 else 'r', alpha=0.5)
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=18)
            axes[i, j].set_xticks(range(len(labels)))
            axes[i, j].set_xticklabels(labels, ha='center', fontsize=14, rotation=35)
            if i == 1:
                axes[i, j].set_xlabel('Sequence Length (AA)', fontsize=14)
            max_y_value = metric.values.max() * 1.10 
            axes[i, j].set_ylim(0, max_y_value)
            axes[i, j].tick_params(axis='y', labelsize=14)  # Increase y-axis tick label size
            for bar in bars:
                height = bar.get_height()
                axes[i, j].annotate(f'{height:.1f}%',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=12)
            if j == 0:
                axes[i, j].set_ylabel(chain_labels[i], fontsize=18)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(gen_dir, "chain_analysis.png"))

# per region stats -----------------------------------------------------------------------------------------------
df = df.dropna(subset=['domain_numbering'])
region_counts = df.apply(count_res_per_region, axis=1).apply(pd.Series)
region_counts.columns = ['lenFR1', 'lenCDR1', 'lenFR2', 'lenCDR2', 'lenFR3', 'lenCDR3', 'lenFR4']
df = pd.concat([df, region_counts], axis=1)

hc_df = df[df['chain'] == 'H']
lc_df = df[df['chain'] == 'L']
for i, frame in enumerate([df[df['chain'] == 'H'], df[df['chain'] == 'L']]):
    ctype = 'H' if i == 0 else 'L'
    color = 'r' if ctype == 'L' else 'b'

    domain_presence = frame[['lenFR1', 'lenCDR1', 'lenFR2', 'lenCDR2', 'lenFR3', 'lenCDR3','lenFR4']].gt(0).mean() * 100

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    bars = domain_presence.plot(kind='bar', ax=axes[0, 0], color=color, edgecolor='black', alpha=0.5)
    axes[0, 0].set_title('Domain Presence (%)', fontsize=16)
    axes[0, 0].set_ylabel('structures containing domain (%)', fontsize=16)
    axes[0, 0].set_yticklabels([f'{int(y)}' for y in axes[0, 0].get_yticks()], fontsize=14)
    axes[0, 0].set_xticklabels(['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4'], ha='center', rotation=15, fontsize=14)
    max_value = domain_presence.max()
    axes[0, 0].set_ylim(0, max_value * 1.19)
    for bar in bars.patches:  # add the value label above each bar
        height = bar.get_height()
        axes[0, 0].annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12, rotation=90)

    titles = ['CDR1', 'CDR2', 'CDR3', 'FR1', 'FR2', 'FR3', 'FR4']
    max_bins = max(frame['len'+title].nunique() for title in titles)
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
        hist = sns.histplot(frame['len'+title], bins=max_bins, ax=axes[row, col], stat='percent', kde=False, color=color, discrete=True, alpha=0.5)
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
        max_value = domain_presence.max()
        axes[row, col].set_ylim(0, max_value * 1.19)
        x_ticks = range(x_axis_bounds[title][0], x_axis_bounds[title][1] + 1)
        axes[row, col].set_xticks(x_ticks)
        axes[row, col].set_xticklabels(x_ticks, ha='center', fontsize=14, rotation=90)
        axes[row, col].set_yticklabels([f'{int(y)}' for y in axes[0, 0].get_yticks()], fontsize=14)
        for bar in hist.patches:
            height = bar.get_height()
            if height > 0:  # Only label bars with height > 0 to avoid clutter
                axes[row, col].annotate(f'{height:.1f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),  # 3 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontsize=12, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(gen_dir, f"region_analysis_{ctype}.png"))

df.to_csv(os.path.join(gen_dir, "successful_anarci_annotation.csv"), index=False)
