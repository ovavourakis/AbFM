For your set of generations (edit appropriately) run:

    ./run_trainset_seq_extract.sh                   # create reference set                                      
    sbatch run_trainset_sample_ab_seqs.sh           # AbMPNN on (equivalent) reference set              (long)    
    python trainset_number_ab_seqs.py (2)           # ANARCI on both sets of reference sequences (2x)               
    sbatch run_humab.sh                             # humab on reference sets (7h for large)     (2x)   (long)
    ./run_fold_seqs.sh                              # re-fold the training subset                       (long)
    
    sbatch sample_ab_seqs.py                        # AbMPNN on generations                                     
    python blobb_check_by_chain.py                  # physical violations                                       
    python plot_ramach.py                           # Ramachandran                                              
    python number_ab_seqs.py                        # ANARCI on generated sequences                             
    python fold_seqs.py                             # re-fold                                                   
    sbatch run_humab.sh                             # humab     
    python sc_rmsd.sh                               # scRMSD vs refolds                                                
