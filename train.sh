#!/bin/ksh
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N trident_job
cd /beegfs/data/work/c-2iia/vb710264/trident_torchmil
module load python
source /beegfs/data/work/c-2iia/vb710264/trident_torchmil/trident_venv/bin/activate
python /beegfs/data/work/c-2iia/vb710264/trident_torchmil/test_slide_ccub.py
