#!/usr/bin/env bash
#PBS -l walltime=06:00:00,mem=100GB,ncpus=1,storage=gdata/xp65+gdata/rd53+scratch/rd53+gdata/su58+scratch/su58+gdata/k10+scratch/k10
#PBS -N Analysis_median_a
#PBS -P rd53
#PBS -q normal
#PBS -o ProjSubmit_Analysis_median_a.outlog
#PBS -e ProjSubmit_Analysis_median_a.errlog

module purge
module use /g/data/xp65/public/modules
module load singularity
module load conda/analysis3

source /g/data/xp65/public/apps/med_conda/etc/profile.d/conda.sh
conda activate /g/data/rd53/oggm_1.6.3

cd /g/data/rd53/wy2165/disequilibrium/code_analysis/
python Calculate_regional_mass_median_a.py > $Analysis_median_a.log
