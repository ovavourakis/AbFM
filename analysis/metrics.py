""" Metrics. """
import mdtraj as md
import numpy as np
from openfold.np import residue_constants
from tmtools import tm_align
import os

def blobb_check(pdb_path, rerun_check=True):
    """
    Run blobb_structure_check on a pdb file and and return problem counts.
    """

    dir = os.path.expanduser(os.path.dirname(pdb_path))
    fname = os.path.basename(pdb_path).split('.')[0]
    if rerun_check:
        os.system(f"check_structure --check_only -i {pdb_path} backbone > {dir}/{fname}_strc_ck.out")

    file = os.path.join(dir, f"{fname}_strc_ck.out")
    if os.path.exists(file):
        with open(file, 'r') as file:
            lines = file.readlines()
    else:
        print(f"File {file} does not exist.")
        return  False

    missing_o = 0
    bb_breaks = 0
    bb_bad_links = 0
    covalent_link_problems = False

    for l in lines:
        words = l.split()
        if len(words) == 0:
            continue
        if words[0] == 'Structure':
            pdb_path = words[1]
        elif words[0] == 'Found':
            if 'Residues with missing backbone atoms' in l:
                missing_o = int(words[1])
            elif 'Backbone breaks' in l:
                bb_breaks = int(words[1])
            elif 'Unexpected backbone links' in l:
                bb_bad_links = True
        elif words[0] == 'Consecutive':
            covalent_link_problems = True
    
    return  missing_o, bb_breaks, bb_bad_links, covalent_link_problems

def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    # https://en.wikipedia.org/wiki/Template_modeling_score
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2 

def calc_mdtraj_metrics(pdb_path):
    # DSSP and radius of gyration
    # https://en.wikipedia.org/wiki/DSSP_(algorithm)

    # TODO: multiple-chain-case: may want to return per-chain
    try:
        traj = md.load(pdb_path)
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == 'C')
        pdb_helix_percent = np.mean(pdb_ss == 'H')
        pdb_strand_percent = np.mean(pdb_ss == 'E')
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent 
        pdb_rg = md.compute_rg(traj)[0] # radius of gyration
    except IndexError as e:
        print('Error in calc_mdtraj_metrics: {}'.format(e))
        pdb_ss_percent = 0.0
        pdb_coil_percent = 0.0
        pdb_helix_percent = 0.0
        pdb_strand_percent = 0.0
        pdb_rg = 0.0
    return {
        'non_coil_percent': pdb_ss_percent, # NOTE: currently monitored during training for checkpointing
        'coil_percent': pdb_coil_percent,
        'helix_percent': pdb_helix_percent,
        'strand_percent': pdb_strand_percent,
        'radius_of_gyration': pdb_rg,
    }

def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):
    
    # TODO: modify for multi-chain case (reutrn per-chain!)

    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + bond_tol))
    
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < clash_tol
    return {
        'ca_ca_deviation': ca_ca_dev,
        'ca_ca_valid_percent': ca_ca_valid,
        'num_ca_ca_clashes': np.sum(clashes),
    }
