import os
import pandas as pd

from anarci import anarci
from Bio import SeqIO

gen_dir = "/vols/opig/users/vavourakis/generations/gpu04_1xA100"
df = pd.DataFrame()


fasta_dir = os.path.join(gen_dir, "designed_seqs/seqs")
all_fastas = [f for f in os.listdir(fasta_dir) if f.endswith('.fa')]
for f in all_fastas:
    with open(f) as handle:
        struc_name = f.split(".")[0]

        seq_id = -1
        for record in SeqIO.parse(handle, "fasta"):
            seq_id += 1

            sequences = [ (f'H', str(record.seq).split("/")[0]), 
                          (f'L', str(record.seq).split("/")[1]) ]
            numbering, alignment_details, hit_tables = anarci(sequences, 
                                                            scheme="imgt", 
                                                            output=False
            )

            for seq in range(len(sequences)):
                intended_chain_type = sequences[seq][0]
                
                numbering_failure = True if numbering[seq] is None else False
                single_domain = False if len(numbering[seq]) > 1 else True
                
                if single_domain:
                    for domain in range(len(numbering[seq])): # should have length 1
                        human = True if alignment_details[seq][domain]['species'] == 'human' else False
                        correct_type = True if alignment_details[seq][domain]['chain_type'] == intended_chain_type else False

                    if human and correct_type:
                        domain_numbering, start_index, end_index = numbering[seq][domain]
                    else:
                        domain_numbering, start_index, end_index = None, None, None
                else:
                    human, correct_type = False, False
                    domain_numbering, start_index, end_index = None, None, None

                df = df.append({    'structure': struc_name,
                                    'seq_id': seq_id,
                                    'chain': intended_chain_type,
                                    'numbering_failure': numbering_failure,
                                    'single_domain': single_domain,
                                    'human': human,
                                    'correct_type': correct_type,
                                    'domain_numbering': domain_numbering,
                                    'start_index': start_index,
                                    'end_index': end_index
                                }, ignore_index=True)