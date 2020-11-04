#!/bin/bash

# task name
#SBATCH -J RuntimeBenchmark

# set time limit
#SBATCH --time=8:00:00
#SBATCH -c 32

# put it on reserved node
#SBATCH --reservation=hpc_10
#SBATCH -w lxbk0721

# choose working directory
#SBATCH -D /lustre/bhs/ccaliari/TorchOcelot/src

# redirect standard out and err
#SBATCH -o ./reports/%x_%j.out.log

# execute
hostname
python RuntimeBenchmark.py
