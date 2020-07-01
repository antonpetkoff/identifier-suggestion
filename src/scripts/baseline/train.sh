#!/bin/bash

PYTHONPATH=. python src/pipelines/baseline.py \
  --experiment_name baseline_experiment
  --dir_data data/processed/subtoken/ \
  --file_checkpoint_dir models/checkpoints/baseline/ \
  --dir_preprocessed_data data/processed/seq2seq/ \
  --max_input_length 128 \
  --max_output_length 8 \
  --input_vocab_size 5000 \
  --input_embedding_dim 50 \
  --output_vocab_size 5000 \
  --output_embedding_dim 50 \
  --latent_dim 320 \
  --learning_rate 0.001 \
  --dropout_rate 0.05 \
  --epochs 1 \
  --early_stopping_patience 3 \
  --early_stopping_min_delta 0.001 \
  --evaluation_dataset validation \
  --batch_size 64 \
  --random_seed 1
