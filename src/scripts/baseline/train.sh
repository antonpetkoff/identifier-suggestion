#!/bin/bash

PYTHONPATH=. python src/pipelines/baseline.py \
  --dir_data data/processed/subtoken/ \
  --file_checkpoint_dir models/checkpoints/baseline/ \
  --dir_preprocessed_data data/processed/seq2seq/ \
  --eval_averaging macro \
  --max_input_length 200 \
  --max_output_length 8 \
  --input_vocab_size 20000 \
  --input_embedding_dim 50 \
  --output_vocab_size 15000 \
  --output_embedding_dim 50 \
  --latent_dim 512 \
  --learning_rate 0.0001 \
  --epochs 1 \
  --batch_size 128 \
  --random_seed 1
