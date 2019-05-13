#!/bin/bash

PYTHONPATH=/users/hdpsbp/bk/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

PATH=/apps/dslab/anaconda/python3/bin:$PATH



for level in {0..5}
do
    echo $level

    nohup python -u  core/train.py train_ex {} [] $level > log/scy_en_level_$level.log 2>&1
done
