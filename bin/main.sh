#!/bin/bash
cd "$(dirname "$0")"


cd ..
export PYTHONPATH=.:/apps/felix/kdd_bd:/home/jovyan/mnt/felix/kdd_bd:$PYTHONPATH

#PATH=/apps/dslab/anaconda/python3/bin:$PATH


python ./core/train.py train_ex $1 $2 $3  #


