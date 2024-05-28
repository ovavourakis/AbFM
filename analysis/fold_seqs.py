import argparse, os
from tqdm import tqdm
from Bio import SeqIO

from ImmuneBuilder import ABodyBuilder2

parser = argparse.ArgumentParser(description='Use ABB2 to forward-fold the sampled sequences for the generated structures.')
parser.add_argument('--gen_dir', type=str, help='Path to directory with generated structures and sequences.', required=True)
args = parser.parse_args()

gen_dir = args.gen_dir
out_dir = os.path.join(gen_dir, "refolded_strucs")

predictor = ABodyBuilder2()

fasta_dir = os.path.join(gen_dir, "designed_seqs/seqs")
all_fastas = [f for f in os.listdir(fasta_dir) if f.endswith('.fa')]
for f in tqdm(all_fastas):
    with open(os.path.join(fasta_dir, f)) as handle:
        struc_name = f.split(".")[0]
        seq_id = -1
        for record in SeqIO.parse(handle, "fasta"):
            seq_id += 1

            outfile = f'{struc_name}_seq_{seq_id}_fold.pdb'
            sequences = {'H': str(record.seq).split("/")[0],
                         'L': str(record.seq).split("/")[1]}

            antibody = predictor.predict(sequences)
            antibody.save(outfile)