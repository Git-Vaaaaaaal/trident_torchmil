#!/bin/ksh
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N mil_train
cd /beegfs/data/work/c-2iia/vb710264/mil_lab
module load python/3.11/anaconda/2024.02
bash
source ~/.bashrc
conda activate trident_env311

cd /beegfs/data/work/c-2iia/vb710264/trident/trident_torchmil
python /beegfs/data/work/c-2iia/vb710264/trident/trident_torchmil/feature_extraction.py
python /beegfs/data/work/c-2iia/vb710264/trident/trident_torchmil/torchmil_process.py
