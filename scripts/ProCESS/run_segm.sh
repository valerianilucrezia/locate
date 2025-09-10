#!/bin/bash
#SBATCH --partition=GENOA
#SBATCH --job-name=seg_ProCESS_locate
#SBATCH --nodes=1
#SBATCH --array=1-30
####SBARCH --tasks-per-node=1
#SBARCH --cpus-per-task=2
#SBATCH --mem 50g
#SBATCH --time=48:00:00
#SBATCH --output=out/seg_%a.out
####SBATCH --output=sim.out

sim=${SLURM_ARRAY_TASK_ID}
echo $sim

export PATH="/orfeo/scratch/area/lvaleriani/myconda/bin:$PATH"
source activate locate

PYTHONPATH="/orfeo/scratch/area/lvaleriani/locate/"
export PYTHONPATH="/orfeo/scratch/area/lvaleriani/locate/"

export PATH="/orfeo/scratch/area/lvaleriani/myconda/envs/locate/bin:/orfeo/scratch/area/lvaleriani/myconda/condabin:/orfeo/scratch/area/lvaleriani/myconda/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/u/area/lvaleriani/.local/bin:/u/area/lvaleriani/bin"

python segment_ProCESS.py -b /orfeo/scratch/area/lvaleriani/utils_locate/simulations_rRACES/test_dr/out/clonal -s sim_${sim}
