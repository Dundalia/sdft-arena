#!/bin/bash

module load anaconda/3 cudatoolkit/12.6.0
conda activate $SCRATCH/arena-capstone/venv
export HF_HOME=$SCRATCH
