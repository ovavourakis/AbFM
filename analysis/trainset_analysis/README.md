Scripts for analysing the training data, to provide comparators to metrics obtained on generated structures and sequences.
Some scripts are called by other, so stick to running (in order):

    sbatch run_trainset_sample_ab_seqs.sh
        or
    run_trainset_seq_extract.sh
        and then
    python trainset_number_ab_seqs.py
