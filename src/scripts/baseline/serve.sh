#!/bin/bash

PYTHONPATH=. python src/server/main.py \
  --file_model_dir models/checkpoints/baseline-300-epochs/ \
  --vocab_path data/interim/preprocessed/
