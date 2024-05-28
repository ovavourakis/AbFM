import yaml
import numpy as np

def ANARCI_to_sequence(anarci_str: str):
    '''Convert ANARCI numbering string of a single chain to 
    list of sequences by region along with end indices of each region.'''

    anarci_dict = yaml.load(anarci_str, Loader=yaml.SafeLoader)
    if 'fwh1' in anarci_dict.keys():
        keys = ['fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4']
    elif 'fwk1' in anarci_dict.keys():
        keys = ['fwk1', 'cdrk1', 'fwk2', 'cdrk2', 'fwk3', 'cdrk3', 'fwk4']
    elif 'fwl1' in anarci_dict.keys():
        keys = ['fwl1', 'cdrl1', 'fwl2', 'cdrl2', 'fwl3', 'cdrl3', 'fwl4']
    else:
        raise ValueError('Cannot parse ANARCI string.')
        
    seqs = []
    for key in keys:
        seqs.append(''.join([i[1] for i in anarci_dict[key].items()]))
    idx = list(np.cumsum([len(seq) for seq in seqs]))

    return seqs, idx

def ANARCIs_to_sequence(anarci_str_h, anarci_str_l):
    '''
    Convert ANARCI numbering strings of heavy and light chains to 
    list of sequences by region along with end-indices by region and
    total sequence length. Light chains start at index 1000.
    
    Returns:
        (fwr1_h, cdr1_h, fwr2_h, cdr2_h, fwr3_h, cdr3_h, fwr4_h,
         fwr1_l, cdr1_l, fwr2_l, cdr2_l, fwr3_l, cdr3_l, fwr4_l,
         full_seq, cdr_concat, fwk_concat, 
         (0, end_fwr1_h, end_cdr1_h, end_fwr2_h, end_cdr2_h, end_fwr3_h, end_cdr3_h, end_fwr4_h,
          1000, end_fwr1_l, end_cdr1_l, end_fwr2_l, end_cdr2_l, end_fwr3_l, end_cdr3_l, end_fwr4_l),
         seq_len
        )
    '''
         
    seqs_h, idx_h = ANARCI_to_sequence(anarci_str_h)
    seqs_l, idx_l = ANARCI_to_sequence(anarci_str_l)
    idx_l = [i+1000 for i in idx_l]

    cdrs = [seqs_h[1] + seqs_h[3] + seqs_h[5] + seqs_l[1] + seqs_l[3] + seqs_l[5]]
    fwks = [seqs_h[0] + seqs_h[2] + seqs_h[4] + seqs_l[0] + seqs_l[2] + seqs_l[4]]
    cdr_concat, fwk_concat, full_seq = [''.join(cdrs)], [''.join(fwks)], [''.join(seqs_h + ['/'] + seqs_l)]

    seqs = seqs_h + seqs_l + full_seq + cdr_concat + fwk_concat
    seq_len = len(full_seq[0])-1 # don't count the slash that separates heavy from light chain
    idx = tuple([0] + idx_h + [1000] + idx_l)
    seqs.extend([idx, seq_len])
    
    return tuple(seqs)