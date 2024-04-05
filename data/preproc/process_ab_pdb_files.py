"""Script for preprocessing PDB files."""

import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
from Bio import PDB
import numpy as np
import mdtraj as md

# TODO: fix imports
from data import utils as du 
from data import parsers
from data import errors


# Define the parser
parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--pdb_dir',
    help='Path to directory with PDB files.',
    type=str)

parser.add_argument(
    '--csv_path',
    help='Path to pre-filtered CSV file with PDB paths and metadata.',
    type=str)

parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=50)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default='preprocessed')
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')


def process_file(file_path: str, write_dir: str):
    # TODO: will have to be adapted for antibodies

    """Processes antibody PDB file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.
        TODO: pickle elements documentation

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propagated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path).replace('.pdb', '')
    metadata['pdb_name'] = pdb_name

    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    # TODO: change this to be relative path from directory root
    metadata['processed_path'] = os.path.abspath(processed_path)
    metadata['raw_path'] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # get chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    
    # count chains
    num_chains = len(struct_chains)
    if num_chains != 2:
        raise errors.DataError(f'Expected 2 chains, got {num_chains} in file {file_path}.')
    metadata['num_chains'] = num_chains

    # ensure chains have correct IDs
    if 'H' not in struct_chains or 'L' not in struct_chains:
        raise errors.DataError(f'Expected chains H and L, got {list(struct_chains.keys())} in file {file_path}.')

    # extract features

    # bb_positions, atom_positions, masks
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # TODO: change for antibodies! - not one protein per chain! re-number second chain

        chain_id = du.chain_str_to_int(chain_id)            # ascii indices of 'H' and 'L'
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # aa sequence and sequence length
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0] # no unnatural AAs
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues') # all-unnatural AAs
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx
    
    # secondary structure and radius of gyration
    try:
        # MDtraj
        traj = md.load(file_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        os.remove(file_path)
    except Exception as e:
        os.remove(file_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]
    
    # Write features to pickles.
    du.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(all_paths, write_dir):
    all_metadata = []
    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            metadata = process_file(
                file_path,
                write_dir)
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')
    return all_metadata


def process_fn(
        file_path,
        verbose=None,
        write_dir=None):
    try:
        start_time = time.time()
        metadata = process_file(
            file_path,
            write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


def main(args):
    pdb_dir = args.pdb_dir

    filtered_csv = pd.read_csv(args.csv_path)
    all_file_paths = filtered_csv['pdb_path'].tolist()
    total_num_paths = len(all_file_paths)

    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths,
            write_dir)
    else:
        # reduce it down to a single-argument function ...
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir)
        # ... which can then be passed to a parallel pool.map()
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, all_file_paths)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)

    # TODO: merge with filtered_csv; filter duplicate columns like pdb_name/oas_id

    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)