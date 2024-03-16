#!/bin/bash

#SBATCH --job-name=t5-training


#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=a40:1 
#SBATCH --time=48:00:00

#SBATCH --chdir=/mmfs1/home/jrl712/amazon_home/nlp_final_project
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/jrl712/amazon_home/nlp_final_project/job_out.txt
#SBATCH --error=/mmfs1/home/jrl712/amazon_home/nlp_final_project/job_err.txt

# Modules to use (optional).
#<e.g., module load apptainer>

# Your programs to run.
/mmfs1/home/jrl712/launch-container-ro.sh /mmfs1/home/jrl712/amazon_home/nlp_final_project/start_train.sh