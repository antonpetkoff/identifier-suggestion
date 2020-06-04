#!/bin/bash

PYTHONPATH=. python src/server/main.py \
  --file_checkpoint_dir models/checkpoints/baseline-30-epochs/ \
  --vocab_path data/interim/preprocessed/
