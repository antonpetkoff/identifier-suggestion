#!/bin/bash

PYTHONPATH=. python src/server/main.py \
  --file_checkpoint_dir models/checkpoints/seq2seq-best/ \
  --vocab_path models/checkpoints/seq2seq-best/
