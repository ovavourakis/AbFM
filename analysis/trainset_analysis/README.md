Scripts for analysing the training data, to provide comparators to metrics obtained on generated structures and sequences.
Some scripts are called by other, so stick to running (in order):

For your set of generations (edit appropriately) run:

    ./run_trainset_seq_extract.sh                   # create reference set                                      x
    sbatch run_trainset_sample_ab_seqs.sh           # AbMPNN on (equivalent) reference set             (long)   * 
    python trainset_number_ab_seqs.py (x2)          # ANARCI on both sets of reference sequences (2x)           x []   
    sbatch run_humab.sh                             # humab on reference sets (7h for large)     (2x)  (long)   x [] 
    
    sbatch sample_ab_seqs.py                        # AbMPNN on generations                                     x
    python blobb_check_by_chain.py                  # physical violations                                       x
    python plot_ramach.py                           # Ramachandran                                              x
    python number_ab_seqs.py                        # ANARCI on generated sequences                             x
    python fold_seqs.py                             # re-fold                                                   x
    sbatch run_humab.sh                             # humab                                                     []
