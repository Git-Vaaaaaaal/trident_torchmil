#!/bin/ksh
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N mil_train
cd /beegfs/data/work/c-2iia/vb710264/mil_lab
module load python
source /beegfs/data/work/c-2iia/vb710264/trident/trident_venv/bin/activate
export PYTHONPATH=/work/c-2iia/vb710264/trident/trident_venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/c-2iia/vb710264/.cache/matplotlib
cd /beegfs/data/work/c-2iia/vb710264/trident/trident_torchmil
python /beegfs/data/work/c-2iia/vb710264/trident/trident_torchmil/feature_extraction.py
python /beegfs/data/work/c-2iia/vb710264/trident/trident_torchmil/torchmil_process.py
