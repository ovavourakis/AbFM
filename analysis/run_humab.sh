#!/bin/bash
#SBATCH -J humab
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=odysseas.vavourakis@balliol.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/vols/opig/users/vavourakis/logs/humab_annotate_trainorig.out
#SBATCH --error=/vols/opig/users/vavourakis/logs/humab_annotate_trainorig.err

# SBATCH --partition=high-opig-test
# SBATCH --clusters=srf_gpu_01
# SBATCH -w nagagpu04.cpu.stats.ox.ac.uk

#SBATCH --partition=interactive-sm-test
#SBATCH --clusters=swan
#SBATCH -w nagagpu06.cpu.stats.ox.ac.uk

gendir='/vols/opig/users/vavourakis/generations/TRAINSET_genseq'    # contains fasta files to be annotated
script_dir='/vols/opig/users/vavourakis/codebase/AbFM/analysis'     # contains humab script to be loaded into container
# script_dir=$(dirname "$(readlink -f "$0")")

source ~/.bashrc


echo "Running humab annotation on $gendir"
singularity exec --bind /vols/opig/users/vavourakis/data/humab_sabbox_data:/sabdab-sabpred/data \
                 --bind $script_dir:/mnt \
                 --bind $gendir:/gendir \
                 /vols/opig/users/vavourakis/bin/sabbox.sif \
                 /bin/bash -c "cd /mnt && python annotate_species_humab.py"

cd $script_dir
conda activate fm

echo "Humab results:"
python plot_humab.py --gen_dir $gen_dir