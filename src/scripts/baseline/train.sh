#!/bin/bash

PYTHONPATH=. python src/pipelines/baseline.py \
  --file_data_raw data/interim/method-names-rich/elasticsearch.csv \
  --file_model_output models/saved/baseline.h5 \
  --max_input_length 100 \
  --max_output_length 5 \
  --input_vocab_size 10000 \
  --input_embedding_dim 50 \
  --output_vocab_size 5000 \
  --output_embedding_dim 50 \
  --latent_dim 128 \
  --learning_rate 0.0001 \
  --epochs 10 \
  --batch_size 128 \
  --random_seed 1
