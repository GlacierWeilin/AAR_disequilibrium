#!/usr/bin/env bash
#PBS -l walltime=48:00:00,mem=100GB,ncpus=48,storage=gdata/xp65+gdata/rd53+scratch/rd53+gdata/su58+scratch/su58+gdata/k10+scratch/k10
#PBS -N Run_disequilibrium
#PBS -P su58
#PBS -q normal
#PBS -o ProjSubmit_Run_disequilibrium.outlog
#PBS -e ProjSubmit_Run_disequilibrium.errlog

module purge
module use /g/data/xp65/public/modules
module load singularity
module load conda/analysis3

source /g/data/xp65/public/apps/med_conda/etc/profile.d/conda.sh
conda activate /g/data/su58/wy2165/envs/PyGEM

cd /g/data/rd53/wy2165/disequilibrium/code_pygem_none/
python Run_disequilibrium.py > $Run_disequilibrium.log
