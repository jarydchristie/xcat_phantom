#!/bin/bash
#SBATCH --account=def-mattonen
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=8   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8G           # memory per node
#SBATCH --time=00-35:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=jchris63@uwo.ca   # Email to which notifications will be $

module load python/3.8
source ~/xcatGANPhantom/bin/activate

python train.py
