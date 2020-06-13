#!/bin/bash

PYTHONPATH=../.. python ../../src/data/preprocess.py \
  --dir_input_data ../interim/repositories/ \
  --dir_output ../processed/ \
  --num_jobs 4 \
  --tokenization_level 'subtoken'
