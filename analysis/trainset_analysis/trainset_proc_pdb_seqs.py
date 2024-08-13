import argparse, os
from analysis.utils import extract_sequences_from_pdbs, renumber_pdbs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDBs for Sequence Analysis.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the folder containing input PDB files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the folder to save output FASTA files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--renumber", action="store_true", help="Re-number residues in PDB files sequentially for each chain.")
    group.add_argument("--extract", action="store_true", help="Extract sequences from PDB files.")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.renumber:
        renumber_pdbs(args.input_path, args.output_path)
    elif args.extract:
        extract_sequences_from_pdbs(args.input_path, args.output_path)