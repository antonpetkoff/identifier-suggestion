#!/bin/bash

PYTHONPATH=. python lib/pipelines/baseline.py \
  --file_data_raw data/method-names-rich/elasticsearch.csv
