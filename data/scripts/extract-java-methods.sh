#!/bin/bash

set -e # Exit immediately if a command errors out

SOURCE_DIR=$1
TARGET_FILE="data/interim/repositories/$(basename ${SOURCE_DIR}).csv"

FILES=$(find "${SOURCE_DIR}" -name '*.java' -type f)

FILE_COUNT=$(echo "${FILES}" | wc -l)

echo "Extracting methods from ${FILE_COUNT} Java source files"

echo "${FILES}" \
| pv -l -s "${FILE_COUNT}" \
| python src/data/extract-java-methods.py > "${TARGET_FILE}"
