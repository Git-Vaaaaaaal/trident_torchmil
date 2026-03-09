#!/bin/ksh
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N mil_train
cd /beegfs/data/work/c-2iia/vb710264/mil_lab
module load python
source /beegfs/data/work/c-2iia/vb710264/mil_lab/milab_venv/bin/activate
export PYTHONPATH=/work/c-2iia/vb710264/mil_lab/milab_venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/c-2iia/vb710264/.cache/matplotlib
cd /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab
python /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab/src/train.py --task coords --wsi_dir ./wsis --job_dir ./trident_processed --mag 20 --patch_size 256 --overlap 0
