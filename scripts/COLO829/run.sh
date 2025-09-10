#!/bin/bash
#SBATCH --partition=GENOA
#SBATCH --job-name=read_colo829
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem 100g
#SBATCH --time=12:00:00
#SBATCH --output=read.out

module load R/4.4.1
Rscript 1_get_data.R 
