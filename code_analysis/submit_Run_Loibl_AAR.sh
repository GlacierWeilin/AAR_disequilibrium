#!/usr/bin/env bash
#PBS -l walltime=12:00:00,mem=100GB,ncpus=48,storage=gdata/xp65+gdata/rd53+scratch/rd53+gdata/su58+scratch/su58+gdata/k10+scratch/k10
#PBS -N Run_Loibl_AAR
#PBS -P rd53
#PBS -q normal
#PBS -o ProjSubmit_Run_Loibl_AAR.outlog
#PBS -e ProjSubmit_Run_Loibl_AAR.errlog

module purge
module use /g/data/xp65/public/modules
module load singularity
module load conda/analysis3

source /g/data/xp65/public/apps/med_conda/etc/profile.d/conda.sh
conda activate /g/data/rd53/oggm_1.6.3

cd /g/data/rd53/wy2165/disequilibrium/code_analysis/
python Run_Loibl_AAR.py > $Run_Loibl_AAR.log
