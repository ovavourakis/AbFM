import numpy as np
import os
import re
from data import protein
from openfold.utils import rigid_utils

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

Rigid = rigid_utils.Rigid

def plot_ramachandran(phi_psi_angles, title, plot_type="kde_fill", ax=None, color="r"):
    # phi_psi_angles is a single list of tuples (phi, psi)
    phi_angles = [phi * 180 / np.pi for phi, psi in phi_psi_angles if phi is not None]  # radians to degrees
    psi_angles = [psi * 180 / np.pi for phi, psi in phi_psi_angles if psi is not None]  # radians to degrees
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if plot_type == "kde_fill":
        sns.kdeplot(x=phi_angles, y=psi_angles, ax=ax, color=color, fill=True, 
                    levels=np.linspace(0.1, 1, 10), cut=0)
    elif plot_type == "kde_line":
        sns.kdeplot(x=phi_angles, y=psi_angles, ax=ax, color=color, fill=False, 
                    levels=np.linspace(0.1, 1, 10), cut=0)
    elif plot_type == "scatter":
        ax.scatter(phi_angles, psi_angles, color=color, alpha=0.2, s=1)
    
    ax.set_xlabel(r'$\phi$ ($^\circ$)', fontsize=12)
    ax.set_ylabel(r'$\psi$ ($^\circ$)', fontsize=12)
    # ax.set_title(title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.legend()

    return ax

def overlay_ramachandran(data, reference, title='Title', color='b', 
                         labels=['Data', 'Reference'], fname='ramachandran.png'):
    ax = plot_ramachandran(reference, title, plot_type="kde_fill", color=color)
    ax = plot_ramachandran(data, title, plot_type="kde_line", color="black", ax=ax)

    kde_legendpatch = mpatches.Patch(color=color, label=labels[1], alpha=0.5)
    kde_legendline = mlines.Line2D([], [], color='black', label=labels[0])
    ax.legend(handles=[kde_legendpatch, kde_legendline], loc='lower right')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        res_idx=None,
        aatype=None,
        b_factors=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]

    if res_idx is None:
        residue_index = np.arange(n)
        chain_index = np.zeros(n)
    else:
        residue_index = res_idx
        chain_index = np.where(residue_index <= 1000, 0, 1)
        residue_index = np.where(residue_index <= 1000, residue_index, residue_index - 1000)
    
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors)


def write_prot_to_pdb(
        prot_pos: np.ndarray,
        file_path: str,
        res_idx: np.ndarray=None,
        aatype: np.ndarray=None,
        overwrite=False,
        no_indexing=False,
        b_factors=None,
    ):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path

    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, res_idx=res_idx, aatype=aatype, b_factors=b_factors)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, res_idx=res_idx, aatype=aatype, b_factors=b_factors)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path
