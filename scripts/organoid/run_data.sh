#!/bin/bash
#SBATCH --partition=GENOA
#SBATCH --job-name=read_organoids
#SBATCH --nodes=1
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem 50g
#SBATCH --time=48:00:00
#SBATCH --output=log/data_%a.out

samples=(PDO61 PDO6 PDO55 PDO74 PDO57_VII PDO57_III)
sample=${samples[$SLURM_ARRAY_TASK_ID]}
echo "Running $sample"

module load R/4.4.1
Rscript 1_get_data.R -s $sample 
