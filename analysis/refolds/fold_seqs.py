"""
Usage:
    python fold_seqs.py --jobindex <int> --gen_dir <path to generated structures and sequences>
    python fold_seqs.py --jobindex 0 --gen_dir /vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun
"""

import argparse, os, time
from Bio import SeqIO
from ImmuneBuilder import ABodyBuilder2

parser = argparse.ArgumentParser(description='Use ABB2 to forward-fold the sampled sequences for the generated structures.')
parser.add_argument('--gen_dir', type=str, help='Path to directory with generated structures and sequences.', required=True)
parser.add_argument('--jobindex', type=int, help='Job index. Job processes index-th batch of sequences from directory.', default=0)
args = parser.parse_args()

# set up directories
gen_dir = args.gen_dir
if os.path.exists(os.path.join(gen_dir,'designed_seqs/seqs')):
    fasta_dir = os.path.join(gen_dir, "designed_seqs/seqs")
else:
    fasta_dir = os.path.join(gen_dir, "seqs")
out_dir = os.path.join(gen_dir, "refolded_strucs")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("Parsing sequences...")
tasks, i = [], args.jobindex
fastas_per_job = 5 if 'designed_seqs' in fasta_dir else 50
all_fastas = [f for i, f in enumerate(sorted(os.listdir(fasta_dir))) if f.endswith('.fa')][i*fastas_per_job:i*fastas_per_job+fastas_per_job]
print(all_fastas)
for f in all_fastas:
    with open(os.path.join(fasta_dir, f)) as handle:
        seq_id = -1
        for record in SeqIO.parse(handle, "fasta"):
            seq_id += 1
            struc_name = f.split(".")[0]
            outfile = os.path.join(out_dir, f'{struc_name}_seq_{seq_id}_fold.pdb')
            if os.path.exists(outfile):
                print(f"{outfile} already exists. Skipping...")
                continue
            sequences = {'H': str(record.seq).split("/")[0], 'L': str(record.seq).split("/")[1]}
            tasks.append((outfile, sequences))

print('Initialising ABB2...')
predictor = ABodyBuilder2()
antibodies = []
print("Folding sequences...")
for task in tasks:
    start_time = time.time()
    outfile, sequences = task
    antibodies.append((outfile, predictor.predict(sequences)))
    print(f"Completed folding for {outfile} in {time.time() - start_time:.2f} seconds")

print("Refining and Saving structures...")
for outfile, antibody in antibodies:
    start_time = time.time()
    antibody.save(outfile)
    print(f"Refined and written {outfile} in {time.time() - start_time:.2f} seconds")