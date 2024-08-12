import os, argparse
from tqdm import tqdm

from analysis.utils import renumber_pdbs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-number residues in PDB files sequentially for each chain.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the folder containing input PDB files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the folder to save output PDB files.")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    renumber_pdbs(args.input_path, args.output_path)
