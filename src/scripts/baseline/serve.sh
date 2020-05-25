#!/bin/bash

PYTHONPATH=. python src/server/main.py \
  --file_checkpoint_dir models/checkpoints/baseline/ \
  --vocab_path data/interim/preprocessed/
