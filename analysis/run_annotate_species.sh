#!/bin/bash
#SBATCH -J species
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
# SBATCH --mem-per-cpu=10G
#SBATCH --mail-user=odysseas.vavourakis@balliol.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/vols/opig/users/vavourakis/logs/species_annotate.out
#SBATCH --error=/vols/opig/users/vavourakis/logs/species_annotate.err

#SBATCH --partition=high-opig-test
#SBATCH --clusters=srf_gpu_01
#SBATCH -w nagagpu04.cpu.stats.ox.ac.uk

# SBATCH --partition=interactive-sm-test
# SBATCH --clusters=swan
# SBATCH -w nagagpu06.cpu.stats.ox.ac.uk

source ~/.bashrc
conda activate biophi

gen_dir="/vols/opig/users/vavourakis/generations/newclust_newsample_newindex_fullrun"
seq_dir="$gen_dir"/designed_seqs
input_dir="$seq_dir"/seqs
output_dir="$seq_dir"/sapiens_fastas
# seq_dir="$gen_dir"
# input_dir="$gen_dir"/seqs
# output_dir="$gen_dir"/sapiens_fastas
mkdir -p "$output_dir"

echo "Preparing OASis Input..."
IFS=$'\n' read -d '' -r -a files < <(find "$input_dir" -name "*.fa")
total_files=${#files[@]}
current_file=0

for input_file in "${files[@]}"; do
    base_name=$(basename "$input_file" .fa)
    output_file="$output_dir/$base_name.fa"

    awk -v output_file="$output_file" -v base_name="$base_name" '
        BEGIN { seq_num = 1 }
        /^>/ {
            getline seq
            split(seq, sequences, "/")
            for (i = 1; i <= length(sequences); i++) {
                header = ">" base_name "_" int((seq_num + 1) / 2)
                print header > output_file
                print sequences[i] > output_file
                seq_num++
            }
        }
    ' "$input_file"

    current_file=$((current_file + 1))
    echo -ne "Progress: $((current_file * 100 / total_files))% ($current_file/$total_files)\r"
done
echo -ne '\n'

echo "Merging OASis Input & Deleting Temporary Files..."
cat "$output_dir"/* > "$seq_dir"/oasis_inputs.fa
rm -rf "$output_dir"

echo "Scoring Humanness with BioPhi OASis..."
biophi oasis "$seq_dir"/oasis_inputs.fa \
        --oasis-db /vols/opig/users/vavourakis/data/biophi/OASis_9mers_v1.db \
        --output "$seq_dir"/oasis_humanness.xlsx \
        --scheme imgt \
        --cdr-definition imgt \
        --min-percent-subjects 1 \
        --summary

conda deactivate

# TODO: plot the results