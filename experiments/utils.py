"""Utility functions for experiments."""
import logging
import torch
import os
import numpy as np
from analysis import utils as au
from pytorch_lightning.utilities.rank_zero import rank_zero_only

def save_traj(
        sample: np.ndarray,         # final state as atom coords
        bb_prot_traj: np.ndarray,   # entire trajectory as atom coords
        x1_traj: np.ndarray,        # trajectory of final-state projections as atom coords
        diffuse_mask: np.ndarray,   # which residues are diffused (N,)
        res_idx: np.ndarray,        # residue index (N,)
        output_dir: str,
        aatype = None,
    ):
    """Writes final sample and CNF trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x1_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x1_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # write sample
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x1_traj_path = os.path.join(output_dir, 'x1_traj.pdb')

    # use b-factors to specify which residues were generated
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
        res_idx=res_idx,
    )
    prot_traj_path = au.write_prot_to_pdb(
        bb_prot_traj,
        prot_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
        res_idx=res_idx,
    )
    x1_traj_path = au.write_prot_to_pdb(
        x1_traj,
        x1_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
        res_idx=res_idx,
    )
    return {
        'sample_path': sample_path,
        'traj_path': prot_traj_path,
        'x1_traj_path': x1_traj_path,
    }


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict):
    """
    Flattens a nested dict into a list of two- or three-tuples, 
    from which a flat dictionary can be reconstructed.
    """
    
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened
