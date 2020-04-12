#!/bin/bash

PYTHONPATH=. python lib/pipelines/char-rnn-baseline.py \
  --file_data_raw data/method-names-rich/elasticsearch.csv \
  --file_model_output models/char-rnn-baseline.h5 \
  --max_input_length 100 \
  --max_output_length 30 \
  --input_vocab_size 600 \
  --input_embedding_dim 50 \
  --output_vocab_size 70 \
  --output_embedding_dim 50 \
  --latent_dim 512 \
  --learning_rate 0.0001 \
  --epochs 10 \
  --batch_size 512 \
  --random_seed 1
