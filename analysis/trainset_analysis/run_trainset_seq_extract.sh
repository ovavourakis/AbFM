source ~/.bashrc
conda activate fm

metadata='/vols/opig/users/vavourakis/data/ab_processed_newclust_newindex/metadata.csv'
folder_with_pdbs='/vols/opig/users/vavourakis/generations/TRAINSET_origseq'
if [ ! -d $folder_with_pdbs ] 
then
    mkdir -p $folder_with_pdbs
fi

echo 'Sub-sampling training set structures...'
python trainset_lencombo_sampler.py --metadata_csv $metadata > $folder_with_pdbs'/random2kpdbs.txt'

echo 'Copying samples...'
IFS=$'\n' read -d '' -r -a files < $folder_with_pdbs'/random2kpdbs.txt'
total_files=${#files[@]}
current_file=0

for file in "${files[@]}"; do
    cp "$file" "$folder_with_pdbs"
    current_file=$((current_file + 1))
    echo -ne "Progress: $((current_file * 100 / total_files))% ($current_file/$total_files)\r"
done
echo -ne '\n'

echo 'Extracting sequences...'
if [ ! -d $folder_with_pdbs'/seqs' ]; then
    mkdir -p $folder_with_pdbs'/seqs'
fi
python trainset_proc_pdb_seqs.py --extract --input_path $folder_with_pdbs --output_path $folder_with_pdbs'/seqs'

echo 'Deleting renamed PDBs...'
find $folder_with_pdbs -type f -name "*.pdb" -exec rm -f {} +

echo 'Done.'