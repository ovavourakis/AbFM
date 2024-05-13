#!/bin/bash
#SBATCH -J ab_seq
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:Ampere_A100_80GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-user=odysseas.vavourakis@balliol.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=interactive-sm-test
#SBATCH --clusters=swan
#SBATCH -w nagagpu06.cpu.stats.ox.ac.uk
#SBATCH --output=/vols/opig/users/vavourakis/logs/pmpnn_ab_seqs.out
#SBATCH --error=/vols/opig/users/vavourakis/logs/pmpnn_ab_seqs.err

source ~/.bashrc
conda activate fm

echo 'Copying and renaming samples...'
generations="/vols/opig/users/vavourakis/generations/gpu04_1xA100"
folder_with_pdbs=$generations"/designed_seqs"
if [ ! -d $folder_with_pdbs ] 
then
    mkdir -p $folder_with_pdbs
fi
find "$generations" -type f -name "sample.pdb" | while read -r file; do
    # Extract the directory name and sample number
    dirName=$(dirname "$(dirname "$file")")      # 'length_215'
    sampleNum=$(basename "$(dirname "$file")")   # 'sample_0'
    newName="${dirName##*/}_${sampleNum}.pdb"    # 'length_215_sample_0.pdb'
    cp "$file" "$folder_with_pdbs/$newName"
done

# run ProteinMPNN to generate sequences
echo 'Running AbMPNN...'
output_dir=$folder_with_pdbs
path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$output_dir"/assigned_pdbs.jsonl"
chains_to_design="H L"

pmpnn_path=/vols/opig/users/vavourakis/codebase/ProteinMPNN
weights_path=/vols/opig/users/vavourakis/weights

echo 'Parsing Multiple Chains...'
python $pmpnn_path"/helper_scripts/parse_multiple_chains.py" \
                                    --input_path=$folder_with_pdbs \
                                    --output_path=$path_for_parsed_chains
echo 'Assigning Fixed Chains...'
python $pmpnn_path"/helper_scripts/assign_fixed_chains.py" \
                                    --input_path=$path_for_parsed_chains \
                                    --output_path=$path_for_assigned_chains \
                                    --chain_list "$chains_to_design"
echo 'Sampling Sequences...'
# batch size must be <= num_seq_per_target, I think 
python $pmpnn_path"/protein_mpnn_run.py" \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --out_folder $output_dir \
        --num_seq_per_target 20 \
        --sampling_temp "0.2" \
        --seed 37 \
        --batch_size 20 \
        --path_to_model_weights $weights_path \
        --model_name "abmpnn"

echo 'Deleting renamed PDBs...'
find $folder_with_pdbs -type f -name "*.pdb" -exec rm -f {} +
echo 'Done.'