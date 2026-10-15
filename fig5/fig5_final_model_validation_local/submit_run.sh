#!/bin/bash
#SBATCH --job-name=run_1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH -t 24:00:00
#SBATCH -o logs/run_1.out
#SBATCH -e logs/run_1.err



# Run your Python script
./run_all.sh \
  --histidine-rdf-dir ~/histidines/histidines_rdfs \
  --control-rdf-dir ~/controls/controls_rdfs
