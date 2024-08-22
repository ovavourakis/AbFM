""" SPACE2 (diversity tool) expects a directory of IMGT-numbered antibody pdbs. 
    Here, we take our sequence-less generated structures and re-index the residues in the pdbs
    based on the IMGT numbering of the inverse-folded sequence with the lowest scRMSD. """

import os
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Model, Chain, Residue, Structure

def renumber_pdb(scaffold_pdb_path, imgt_pdb_path, outdir):
    parser = PDBParser(QUIET=True)
    # get structures
    scaffold_structure = parser.get_structure('scaffold', scaffold_pdb_path)
    imgt_structure = parser.get_structure('imgt', imgt_pdb_path)
    
    new_structure = Structure.Structure('new_structure')  # create a new empty structure
    for scaffold_model, imgt_model in zip(scaffold_structure, imgt_structure):
        new_model = Model.Model(scaffold_model.id)
        for scaffold_chain, imgt_chain in zip(scaffold_model, imgt_model):
            new_chain = Chain.Chain(scaffold_chain.id)
            for scaffold_residue, imgt_residue in zip(scaffold_chain, imgt_chain):
                new_residue = Residue.Residue(imgt_residue.id, scaffold_residue.resname, scaffold_residue.segid)
                for atom in scaffold_residue:
                    new_residue.add(atom.copy())
                new_chain.add(new_residue)
            new_model.add(new_chain)
        new_structure.add(new_model)
    
    io = PDBIO()
    io.set_structure(new_structure)
    struc = '_'.join(os.path.basename(imgt_pdb_path).split('_')[:-1] + ['gen.pdb'])
    output_path = os.path.join(outdir, struc)
    io.save(output_path, preserve_atom_numbering=True)


gen_dir = '/vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun'

out_dir = os.path.join(gen_dir, 'renumbered_pdbs')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
best_rmsds_path = os.path.join(gen_dir, 'designed_seqs', 'best_rmsds.csv')
best_rmsds = pd.read_csv(best_rmsds_path)

print('IMGT-numbering generated PDBs based on best re-folded structures...')
tqdm.pandas(desc="Renumbering PDBs")
best_rmsds.progress_apply(lambda row: renumber_pdb(row['scaffold_path'], row['refolded_path'], out_dir), axis=1)