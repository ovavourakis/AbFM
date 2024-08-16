'''
Hu-mAb species annotation script to be run inside Sabbox Singularity container. 
Called from sbatch script.
'''

import os, pickle, warnings
from Bio import SeqIO
import anarci
from humab.prettifying import prettify_scores
from humab.humanisation import encode_sequence_for_rf
from humab.constants import RF_THRESHOLDS, RF_LIST

def load_rf(v_gene):
    """Load Random Forest models from pickle files."""
    rf_path = os.path.join(rf_dir, v_gene+".pkl")
    warnings.filterwarnings("ignore") # suppress warnings by RandomForestClassifier
    rf_model = pickle.load(open(rf_path, 'rb'))
    return rf_model

def score_humanness(sequence, rf_model):
    """Use a Random Forest model to obtain the humanness score for a sequence."""
   
    seq_ohe = encode_sequence_for_rf(sequence)
    score = rf_model.predict_proba(seq_ohe)

    return score[0][0][1]

def score(sequence, user_chain_type, verbose=False):
    """
    Function that takes a sequence and calculates its humanness score.

    Args:
        sequence: the sequence to score.
        user_chain_type: heavy/light depending on the user inputs.
        verbose: if True, each step of the process will be logged to the 
            terminal.

    Returns:
        A dictionary containing the scores according to each relevant
        RF model and whether the sequence would be classified as human
        (True/False).
    """

    # Number input sequence
    try:
        if verbose:
            print("Calculating humanness for input %s chain.\n" %(user_chain_type))
            print("Numbering input sequence with ANARCI...\n")
        numbered_sequence, chain_type = anarci.number(sequence)
    except:
        raise

    # Check ANARCI could number sequence, if not stop
    if not numbered_sequence:
        print("ERROR: ANARCI was not able to number your sequence.")
        return

    if chain_type != user_chain_type:
        print("ERROR: ANARCI-determined chain type does not match what user specified. Please check your inputs.")
        return

    # Remove 'empty' residues
    numbered_sequence = [r for r in numbered_sequence if r[1] != "-"]

    # Select Random Forest models
    if chain_type == "H":
        to_use = [rf for rf in RF_LIST if rf.startswith("h")]
        is_kappa = False
    elif chain_type == "L":
        seq_ohe = encode_sequence_for_rf(numbered_sequence)
        kappa_lambda = kappa_lambda_classifier.predict(seq_ohe)
        is_kappa = kappa_lambda[0][1] == 1 # If kappa
        if is_kappa: # Kappa
            if verbose:
                print('Light chain sequence classified as kappa')
            to_use = [rf for rf in RF_LIST if rf.startswith("k")]
        else: # lambda
            if verbose:
                print('Light chain sequence classified as lambda')
            to_use = [rf for rf in RF_LIST if rf.startswith("l")]

    # Score sequence
    if verbose:
        print("Scoring humanness of sequence using V-genes %s\n" %(", ".join(to_use)))

    score_dict = {}
    for v_gene in to_use:
        rf_model = globals()[f'{v_gene}_rf']
        s = score_humanness(numbered_sequence, rf_model)
        # print(v_gene, s)
        human = s >= RF_THRESHOLDS[v_gene]
        score_dict[v_gene] = {"score": s, "threshold": RF_THRESHOLDS[v_gene], "classified_human": str(human)}

    return score_dict, is_kappa

# globals
rf_dir = '/sabdab-sabpred/data/humab_models/'
try:
    print(f'Loading kappa lambda classifier model...')
    with open(f"{rf_dir}/kappa_lambda_classifier_100.pkl","rb") as f:
        kappa_lambda_classifier = pickle.load(f)
except OSError as e:
    print('Kappa lambda classifier model file not available!')

for v_gene in RF_LIST:
    try:
        print(f'Loading {v_gene} model...')
        globals()[f'{v_gene}_rf'] = load_rf(v_gene)
    except OSError as e:
        print(f'{v_gene} model file not available!')

# parse fastas
print('\nParsing FASTAs...')
seq_dir = '/gendir/seqs'
# seq_dir = '/gendir/designed_seqs/seqs'
tasks, all_fastas = [], [f for f in os.listdir(seq_dir) if f.endswith('.fa')]

for f in all_fastas:
    with open(os.path.join(seq_dir, f)) as handle:
            struc_name = f.split(".")[0]
            seq_id = -1
            for record in SeqIO.parse(handle, "fasta"):
                seq_id += 1
                sequences = [ (f'H', str(record.seq).split("/")[0]), 
                            (f'L', str(record.seq).split("/")[1]) ]
                tasks.append((struc_name, seq_id, sequences))    

# score sequences for humanness
output_rows = ['pdb_id,seq_id,chain,is_kappa,seq_len,len_tot,human']
for i, (struc_name, seq_id, sequences) in enumerate(tasks):
    print(f'Annotating Sequences...{i/len(tasks)*100:.1f}%', end='\r')
    for seq in range(len(sequences)):
        # data for later stratification
        intended_chain_type = sequences[seq][0]               # H or L
        seq_len = len(sequences[seq][1])                      # length of this chain
        len_tot = len(sequences[0][1]) + len(sequences[1][1]) # length of entire antibody
        # check if human
        scores, is_kappa = score(sequences[seq][1], intended_chain_type, verbose=False)
        output_string = prettify_scores(scores)
        is_human = any(line.strip().endswith('HUMAN') and not line.strip().endswith('NOT HUMAN') for line in output_string.split('\n'))
        # format output
        row = f'{struc_name},{seq_id},{intended_chain_type},{is_kappa},{seq_len},{len_tot},{is_human}'
        output_rows.append(row)

# write to file    
with open('/gendir/humab_annotation.csv', 'w') as f:
    f.write('\n'.join(output_rows))

print('\nDone!')