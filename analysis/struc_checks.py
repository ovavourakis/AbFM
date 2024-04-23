import os

def blobb_check(pdb_path, rerun_check=False):
    """
    Run blobb_structure_check on a pdb file and check if the structure is ok.
    Returns bool_ok.
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

    for l in lines:
        words = l.split()
        if len(words) == 0:
            continue
        if words[0] == 'Structure':
            pdb_path = words[1]
        elif words[0] == 'Found':
            if words[1].isdigit() or words[1]=='Unexpected':
                return  False
        elif words[0] == 'Consecutive':
            return  False
    
    return  True