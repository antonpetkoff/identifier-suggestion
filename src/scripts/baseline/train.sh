#!/bin/bash

PYTHONPATH=. python src/pipelines/baseline.py \
  --file_data_raw data/interim/method-names-rich/elasticsearch.csv \
  --file_model_output models/saved/baseline.h5 \
  --dir_preprocessed_data data/interim/preprocessed/ \
  --max_input_length 200 \
  --max_output_length 8 \
  --input_vocab_size 20000 \
  --input_embedding_dim 128 \
  --output_vocab_size 6000 \
  --output_embedding_dim 128 \
  --latent_dim 512 \
  --learning_rate 0.0001 \
  --epochs 10 \
  --batch_size 64 \
  --random_seed 1
