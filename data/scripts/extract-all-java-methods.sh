#!/bin/bash

set -e # Exit immediately if a command errors out

if [ -z $1 ]; then
  echo "First argument should be the number of jobs, e.g. 2, 4, 6, 8, etc."
  exit 1
fi

NUM_JOBS=$1
REPOS_DIR=../external/repositories

echo "Extracting java methods from repositories located in ${REPOS_DIR}"

echo "Using ${NUM_JOBS} parallel jobs"

find "${REPOS_DIR}" -maxdepth 1 -mindepth 1 -type d \
  | parallel -n 1 -j "${NUM_JOBS}" ./extract-java-methods.sh
