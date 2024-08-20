import ast
import numpy as np
from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer

def numbering_to_region_index(anarci_string, seq, chain_type, use_north=True):
    """Convert ANARCI IMGT-numbering string of sequence to region_identifier -> sequential_sequence_index dictionary."""
    imgt_to_region = { # IMGT scheme -> (North et al, 2011) definitions
        'H': {i: region for region, (start, end) in [('FR1_H',(1,23)), ('CDR1_H',(24,40)), ('FR2_H',(41,54)), ('CDR2_H',(55,66)), ('FR3_H',(67,104)), ('CDR3_H',(105,117)), ('FR4_H',(118,128))] for i in range(start, end+1)},
        'L': {i: region for region, (start, end) in [('FR1_L',(1,23)), ('CDR1_L',(24,40)), ('FR2_L',(41,54)), ('CDR2_L',(55,69)), ('FR3_L',(70,104)), ('CDR3_L',(105,117)), ('FR4_L',(118,128))] for i in range(start, end+1)}
    } if use_north else { # IMGT scheme -> IMGT definitions
        i: region for region, (start, end) in [('FR1',(1,26)), ('CDR1',(27,38)), ('FR2',(39,55)), ('CDR2',(56,65)), ('FR3',(66,104)), ('CDR3',(105,117)), ('FR4',(118,128))] for i in range(start, end+1)
    }

    anarci_list = ast.literal_eval(anarci_string)
    region_to_index, seq_idx = {}, 0
    for (pos, _), aa in anarci_list:
        assert aa == seq[seq_idx] or aa == '-', f'Mismatch at position {pos}: {aa} != {seq[seq_idx]}'
        if aa != '-':
            region_to_index.setdefault(imgt_to_region[chain_type][pos], []).append(seq_idx)
            seq_idx += 1
    for idx_list in region_to_index.values():
        assert all(idx_list[i] + 1 == idx_list[i + 1] for i in range(len(idx_list) - 1)), f"Index list {idx_list} is not sequential"

    return region_to_index

def get_backbone_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    backbone_atoms = {'N', 'CA', 'C', 'O'}
    coordinates = [atom.get_coord() for model in structure for chain in model for residue in chain for atom in residue if atom.get_name() in backbone_atoms]
    return np.array(coordinates)

def superpose_strucs(coords1, coords2):
    sup = SVDSuperimposer()
    sup.set(coords1, coords2)
    sup.run()
    return sup, sup.get_transformed()

def get_rmsd(coords1, superposed_coords2):
    return np.sqrt(np.mean(np.linalg.norm(coords1 - superposed_coords2, axis=1)**2))