import numpy as np
import os, re, warnings, ast
import pandas as pd

from tqdm import tqdm
from data import protein
from openfold.utils import rigid_utils
from analysis.refolds.utils import numbering_to_region_index

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

Rigid = rigid_utils.Rigid

from Bio import PDB
from Bio.PDB import Chain, Residue
from Bio.PDB.PDBExceptions import PDBConstructionWarning

def find_closest_len_combos(df, combo):
    len_h, len_l = combo
    closest_combos = []
    for i, length in enumerate([len_h, len_l]):
        subset_df = df[df['len_combo'].apply(lambda x: x[i] == length)]
        closest_length = subset_df.iloc[(subset_df['len_combo'].apply(lambda x: abs(x[1-i] - combo[1-i]))).argmin()]['len_combo'][1-i]
        closest_combos.append((length, closest_length) if i == 0 else (closest_length, length))
    return tuple(closest_combos)

def extract_cdrh3_anarci_strings(anarci_string_dict):
    d = ast.literal_eval(anarci_string_dict)
    l = []
    for k, local_numbering in d.items():
        if k in ['fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4']:
            l.extend([((int(pos[:-1]) if pos[-1].isalpha() else int(pos), pos[-1] if pos[-1].isalpha() else ' '), aa) for pos, aa in local_numbering.items()])
    return str(l)

def sample_equivalent_trainset_lencdrh3(gen_dir, 
                                        trainset_metadata_csv='/vols/opig/users/vavourakis/data/new_OAS_models/OAS_paired_filtered_newclust.csv', 
                                        factor=1):
    """Samples structures (pdb filepaths) from the trainset with equivalent CDRH3 lengths to the generated structures."""
    
    # these filenames indentify which sequence was the best for each generation
    renum_pdbs = [f for f in os.listdir(os.path.join(gen_dir, 'renumbered_pdbs')) if f.endswith('.pdb')] 

    # get cdrh3 lens present in generations
    anarci_df = pd.read_csv(os.path.join(gen_dir, 'designed_seqs', 'anarci_annotation.csv'))
    cdrh3_lens = []
    for index, row in anarci_df.iloc[::2].iterrows(): # skipping light chains
        struc, seq_id = row['structure'], row['seq_id']
        equiv_filename = f"{struc}_seq_{seq_id}_gen.pdb"
        if equiv_filename in renum_pdbs:
            region_index = numbering_to_region_index(row['domain_numbering'], row['seq'], row['chain'], use_north=True)
            cdr3_h_length = len(region_index['CDR3_H']) if 'CDR3_H' in region_index else 0
            if cdr3_h_length != 0:
                cdrh3_lens.append(cdr3_h_length)
    cdrh3_len_counts = pd.Series(cdrh3_lens).value_counts().to_dict()

    # read in trainset metadata
    train_metadata = pd.read_csv(trainset_metadata_csv)
    train_metadata[['VH_seq', 'VL_seq']] = train_metadata['full_seq'].str.split('/', expand=True)
    train_metadata['cdrh3_anarci_strings'] = train_metadata['ANARCI_numbering_heavy'].apply(extract_cdrh3_anarci_strings)
    region_indices = train_metadata.apply(lambda x: numbering_to_region_index(x['cdrh3_anarci_strings'], x['VH_seq'], 'H', use_north=True), axis=1)
    train_metadata['cdr3_h_len'] = region_indices.apply(lambda x: len(x['CDR3_H']) if 'CDR3_H' in x else 0)

    # sample trainset structures with same cdrh3 length distribution
    trainset_paths = []
    for cdrh3_len, count in cdrh3_len_counts.items():
        pathlist = train_metadata[train_metadata['cdr3_h_len'] == cdrh3_len]['pdb_path'].sample(n=count*factor).tolist()
        assert len(pathlist) == count*factor, f"Not enough structures found for this cdrh3 length: {cdrh3_len}."
        trainset_paths.extend(pathlist)

    return trainset_paths


def sample_equivalent_trainset_chainlen(gen_pdb_path_list, trainset_metadata_csv, factor=10):
    """Samples structures (pdb filepaths) from the trainset with equivalent VH/VL 
    lengths to the generated structures."""

    # get combinations present in generations
    all_combos = []
    for pdb_file in tqdm(gen_pdb_path_list):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            structure = PDB.PDBParser().get_structure('structure', pdb_file)
            assert len(structure) == 1, "Multiple models in PDB file."
            model = structure[0]
            assert len(model) == 2, "More than two chains in PDB file."
            lens = tuple([len([res for res in chain if res.id[0] == ' ']) for chain in model])
            all_combos.append(lens)

    # read in trainset metadata
    df = pd.read_csv(trainset_metadata_csv)
    seqs = df['full_seq'].str.split('/', expand=True)
    df['len_combo'] = list(zip(seqs[0].str.len(), seqs[1].str.len()))

    # find trainset structures with equivalent VH/VL lengths
    trainset_paths = []
    for combo in all_combos:
        pathlist = df[(df['len_combo'] == combo) & (~df['raw_path'].isin(trainset_paths))]['raw_path'].sample(n=factor).tolist()
        assert len(pathlist) == factor, "Not enough structures found for this length combination."
        trainset_paths.extend(pathlist)
                 
    return trainset_paths
            
def get_valid_lens_probs(df, column='len_h'):
        """ Take lengths of individual chain (VH or VL) in reference dataset.
        Return lengths within the middle 95% of that distribution and their probabilities. """
        lengths, counts = np.unique(df[column], return_counts=True)
        probabilities = counts / counts.sum()

        cumulative_probabilities = np.cumsum(probabilities)
        valid_mask = (cumulative_probabilities >= 0.025) & (cumulative_probabilities <= 0.975)

        lengths = lengths[valid_mask]
        probabilities = probabilities[valid_mask]/probabilities[valid_mask].sum()

        return lengths, probabilities

def extract_sequences_from_pdb(pdb_file):
    
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
              'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
              'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
              'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
    }
     
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file, pdb_file)
    assert len(structure) == 1, "Multiple models in PDB file."

    seqs = []
    for model in structure:
        for chain in model:
            seq = []
            for res in chain:
                if res.get_id()[0] == ' ':
                    seq.append(d3to1[res.resname])
            seqs.append("".join(seq))
    assert len(seqs) == 2, "More than two chains in PDB file."

    return '/'.join(seqs), os.path.basename(pdb_file).split(".")[0]

def save_sequence_to_fasta(seq, pdb_id, output_path):
    with open(os.path.join(output_path, f"{pdb_id}.fa"), 'w') as f:
        f.write(f">{pdb_id}\n")
        f.write("".join(seq))

def extract_sequences_from_pdbs(input_path, output_path):
    all_pdbs = [f for f in os.listdir(input_path) if f.endswith(".pdb")]
    for pdb_file in tqdm(all_pdbs):
        if pdb_file.endswith(".pdb"):
            seq, pdb_id = extract_sequences_from_pdb(os.path.join(input_path, pdb_file))
            save_sequence_to_fasta(seq, pdb_id, output_path)

def renumber_pdbs(input_path, output_path):
    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()

    for pdb_file in tqdm(os.listdir(input_path)):
        if pdb_file.endswith(".pdb"):

            structure = parser.get_structure(pdb_file, os.path.join(input_path, pdb_file))
            struct_chains = {chain.id.upper() : chain for chain in structure.get_chains()}

            # re-index residues sequentially
            for chain_id in ['H', 'L']:
                chain = struct_chains[chain_id]
                new_chain = Chain.Chain(chain_id)
                for i, res in enumerate(chain, start=1):
                    new_id = (res.id[0], i, res.id[2])
                    new_res = Residue.Residue(new_id, res.resname, res.segid)
                    for atom in res:
                        new_res.add(atom)
                    new_chain.add(new_res)
                struct_chains[chain_id] = new_chain

            # overwrite original pdb file
            new_structure = PDB.Structure.Structure(structure.id)
            new_model = PDB.Model.Model(structure[0].id)
            new_structure.add(new_model)
            
            for chain_id, new_chain in struct_chains.items():
                new_model.add(new_chain)
            
            output_file = os.path.join(output_path, pdb_file)
            io.set_structure(new_structure)
            io.save(output_file)

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
