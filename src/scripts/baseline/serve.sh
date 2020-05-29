#!/bin/bash

PYTHONPATH=. python src/server/main.py \
  --file_checkpoint_dir models/checkpoints/baseline-25-epochs/ \
  --vocab_path data/interim/preprocessed/
