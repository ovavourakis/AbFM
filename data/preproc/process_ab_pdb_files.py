"""
Script for preprocessing antibody PDB files, each of which
should contain just two chains (heavy and light), denoted H and L
(Fv only). The chains are merged into a single chain 'M' (merged).

Requires a pre-filtered CSV file with PDB paths and metadata
(produced by pre_filter_ab_data.py).

Usage:
python process_ab_pdb_files.py --pdb_dir /vols/opig/users/vavourakis/data/OAS_models/structures --csv_path /vols/opig/users/vavourakis/data/OAS_models/OAS_paired_filtered_newclust.csv --num_processes 8 --write_dir /vols/opig/users/vavourakis/data/ab_processed_newclust
"""

import argparse
import dataclasses
import functools as fn
import pandas as pd
import os, warnings
import multiprocessing as mp
import time
from Bio import PDB
from Bio.PDB import Chain, Residue
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import numpy as np
import mdtraj as md

from data import utils as du 
from data import parsers
from data import errors
from analysis.metrics import blobb_check, calculate_phi_psi_angles

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
parser.add_argument(
    '--rerun_blobb',
    help='Whether to re-run blobb_structure_check on every structure.',
    action='store_true')


def process_file(file_path: str, write_dir: str, rerun_blobb: bool = False):
    """
    Processes antibody PDB file into usable, smaller pickles.
    Notably combines heavy and light chains into a single chain called 'M'
    (merged).

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves extracted protein features to pickled dict and returns metadata.
        Pickle contains the following keys:
            - atom_positions: np.ndarray of shape (N, 37, 3) with atom positions.
            - aa_type: np.ndarray of shape (N,) with amino acid types for both 
                chains (numeric; heavy chain first).
            - atom_mask: np.ndarray of shape (N, 37) with atom masks.
            - residue_index: np.ndarray of shape (N,) with residue indices 
                (heavy chain first, light chain index skips to 1001).
            - chain_index: np.ndarray of shape (N,) with chain indices.
                (should contain only '38' repeated N times, the encoding of 'M',
                as per data.utils.chain_str_to_int())
            - b_factors: np.ndarray of shape (N, 37) with B-factors.
            - bb_mask: np.ndarray of shape (N,) with backbone masks. 
                (whether to inlcude the C-alpha of this residue)
            - bb_positions: np.ndarray of shape (N, 3) with backbone C-alpha positions.
            - modeled_idx: np.ndarray of shape (M,) with indices of modeled residues.
                (does NOT contain index skip for start of light chain)
        Metadata contains the following columns:
            - all columns in the input csv, except oas_id and seqlen
            - pdb_name: name of the PDB file
            - processed_path: path to the pickled file
            - raw_path: path to the raw PDB file
            - num_chains: number of chains in the unprocessed PDB file (should be 2)
            - seq_len: total sequence length of the complex (heavy+light chain)
                (we enforce agreement with seqlen in input csv)
            - modeled_seq_len: sequence length of the complex with only modeled residues
                (should hopefully agree with seq_len)
            - coil_percent: percent of residues in coil conformation
            - helix_percent: percent of residues in helix conformation
            - strand_percent: percent of residues in strand conformation

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propagated.
    """

    # sanity-check structure
    checks = blobb_check(file_path, rerun_check=rerun_blobb)
    bb_breaks, bb_bad_links = checks[1], checks[2]
    if bb_breaks > 0 or bb_bad_links:
        print(f'Backbone breaks or crosslinks detected. Skipping file {file_path}.')
        return None, None, None
    # structure seems fine; keep going
    else:
        metadata = {}
        pdb_name = os.path.basename(file_path).replace('.pdb', '')
        metadata['pdb_name'] = pdb_name

        processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
        metadata['processed_path'] = os.path.abspath(processed_path)        # TODO: change this to be relative path from directory root
        metadata['raw_path'] = file_path

        with warnings.catch_warnings():

            # load structure
            warnings.simplefilter('ignore', PDBConstructionWarning)
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_name, file_path)

            # count chains
            struct_chains = {chain.id.upper() : chain for chain in structure.get_chains()}
            num_chains = len(struct_chains)
            if num_chains != 2:
                raise errors.DataError(f'Expected 2 chains, got {num_chains} in file {file_path}.')
            metadata['num_chains'] = num_chains
            
            # ensure chains have correct IDs
            if 'H' not in struct_chains or 'L' not in struct_chains:
                raise errors.DataError(f'Expected chains H and L, got {list(struct_chains.keys())} in file {file_path}.')

            # get phi/psi angles by chain
            vh_angles, vl_angles = calculate_phi_psi_angles(structure)

            # offset light-chain indices by 1000                                                # NOTE: index definition
            modified_light_chain = Chain.Chain("L")
            offset = 1000
            for res in struct_chains['L']:
                new_id = (res.id[0], res.id[1] + offset, res.id[2])
                modified_res = Residue.Residue(new_id, res.resname, res.segid)
                for atom in res:
                    modified_res.add(atom)
                modified_light_chain.add(modified_res)
            
            # merge modified light chain and heavy chain into a single chain
            merged_chain = Chain.Chain("M")
            for chain in [struct_chains['H'], modified_light_chain]:
                for res in chain:
                    merged_chain.add(res)
            struct_chains = {"M": merged_chain}
        
            # Extract Features
            # bb_positions, atom_positions, masks
            struct_feats = []
            all_seqs = set()
            for chain_id, chain in struct_chains.items():
                chain_id = du.chain_str_to_int(chain_id) # ascii index of 'M'
                chain_prot = parsers.process_chain(chain, chain_id)                             # NOTE: index definition
                chain_dict = dataclasses.asdict(chain_prot)
                chain_dict = du.parse_chain_feats(chain_dict)
                all_seqs.add(tuple(chain_dict['aatype']))
                struct_feats.append(chain_dict)
            if len(all_seqs) != 1:
                raise errors.DataError(f'Failed to combine chains for file {file_path}.')
            complex_feats = du.concat_np_features(struct_feats, False)

            # aa sequence and sequence length
            complex_aatype = complex_feats['aatype']
            metadata['seq_len'] = len(complex_aatype)
            modeled_idx = np.where(complex_aatype != 20)[0] # no unnatural AAs
            if np.sum(complex_aatype != 20) == 0:  # all-unnatural AAs
                raise errors.LengthError('Protein contains no residues that are modelled (natural AAs).')
            min_modeled_idx, max_modeled_idx = np.min(modeled_idx), np.max(modeled_idx)
            metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1                 # NOTE: index definition
            complex_feats['modeled_idx'] = modeled_idx                                          # NOTE: index definition
            
            # # secondary structure and radius of gyration
            # try:
            #     # MDtraj
            #     traj = md.load(file_path)
            #     # SS calculation
            #     pdb_ss = md.compute_dssp(traj, simplified=True)
            #     # DG calculation
            #     pdb_dg = md.compute_rg(traj)
            #     # os.remove(file_path)
            # except Exception as e:
            #     # os.remove(file_path)
            #     raise errors.DataError(f'Mdtraj failed with error {e}')

            # chain_dict['ss'] = pdb_ss[0]
            # metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
            # metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
            # metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

            # # Radius of gyration
            # metadata['radius_gyration'] = pdb_dg[0]
            
            # Write features to pickles.
            du.write_pkl(processed_path, complex_feats)

            # Return metadata
            return metadata, vh_angles, vl_angles

def process_serially(all_paths, write_dir, rerun_blobb):
    all_metadata = []
    all_vh_angles = []
    all_vl_angles = []
    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()

            metadata, vh_angles, vl_angles = process_file(file_path, write_dir, 
                                                          rerun_blobb=rerun_blobb)
            
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
            all_vh_angles.extend(vh_angles)
            all_vl_angles.extend(vl_angles)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')
    return all_metadata, all_vh_angles, all_vl_angles


def process_fn(
        file_path,
        verbose=None,
        write_dir=None,
        rerun_blobb=False):
    try:
        start_time = time.time()

        metadata, vh_angles, vl_angles = process_file(file_path, write_dir, 
                                                      rerun_blobb=rerun_blobb)
        
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata, vh_angles, vl_angles
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


def main(args):
    pdb_dir = args.pdb_dir
    rerun_blobb = args.rerun_blobb

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
    print(f'\nFiles will be written to {write_dir}.\n')

    # process each pdb file
    if args.num_processes == 1 or args.debug:
        all_metadata, all_vh_angles, all_vl_angles = process_serially(all_file_paths, 
                                                                        write_dir, 
                                                                        rerun_blobb)
    else:
        # reduce it down to a single-argument function ...
        _process_fn = fn.partial(process_fn, 
                                 verbose=args.verbose, 
                                 write_dir=write_dir,
                                 rerun_blobb=rerun_blobb
        )

        # ... which can then be passed to a parallel pool.map()
        with mp.Pool(processes=args.num_processes) as pool:
            results = pool.map(_process_fn, all_file_paths)
        all_metadata, all_vh_angles, all_vl_angles = zip(*results)

        all_metadata = [x for x in all_metadata if x is not None]
        all_vh_angles = [resangle for chain in all_vh_angles for resangle in chain if resangle[0] is not None and resangle[1] is not None]
        all_vl_angles = [resangle for chain in all_vl_angles for resangle in chain if resangle[0] is not None and resangle[1] is not None]

    # output metadata csv
    metadata_df = pd.DataFrame(all_metadata)
    merged_df = pd.merge(metadata_df, filtered_csv, 
                         left_on='pdb_name', right_on='oas_id', how='inner')
    assert all(merged_df['pdb_name'] == merged_df['oas_id']), "Entries in 'pdb_name' and 'oas_id' do not match."
    assert all(merged_df['seqlen'] == merged_df['seq_len']), "The 'seqlen' and 'seq_len' columns do not match."
    merged_df.drop(columns=['oas_id', 'seqlen'], inplace=True)

    merged_df.to_csv(metadata_path, index=False)
    succeeded = len(merged_df)
    print(f'Finished processing {succeeded}/{total_num_paths} files.')

    # save calculated phi/psi angles to file
    vh_angles_df = pd.DataFrame(all_vh_angles, columns=['phi', 'psi'])
    vl_angles_df = pd.DataFrame(all_vl_angles, columns=['phi', 'psi'])
    vh_angles_path = os.path.join(write_dir, 'vh_angles_train_val_test.csv')
    vl_angles_path = os.path.join(write_dir, 'vl_angles_train_val_test.csv')
    vh_angles_df.to_csv(vh_angles_path, index=False)
    vl_angles_df.to_csv(vl_angles_path, index=False)
    print(f'VH angles saved to {vh_angles_path}')
    print(f'VL angles saved to {vl_angles_path}')

if __name__ == "__main__":
    # don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)