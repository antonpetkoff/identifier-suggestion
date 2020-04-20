#!/bin/bash

# TODO: should we use pipenv scripts?

FILES=$(
  find data/repos/elasticsearch-master/ \
  -name '*.java' \
  -type f
)

FILE_COUNT=$(echo "${FILES}" | wc -l)

echo "${FILE_COUNT}"

echo "${FILES}" \
| pv -l -s "${FILE_COUNT}" \
| python lib/tools/extract-java-methods.py > data/method-names-rich/elasticsearch-ASDFASDF.csv
