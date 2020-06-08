#!/bin/bash

set -e # Exit immediately if a command errors out

NUM_JOBS=8
REPOS_DIR=../external/repositories

echo "Extracting java methods from repositories located in ${REPOS_DIR}"

echo "Using ${NUM_JOBS} parallel jobs"

find "${REPOS_DIR}" -maxdepth 1 -mindepth 1 -type d \
  | parallel -n 1 -j "${NUM_JOBS}" ./extract-java-methods.sh
