#!/bin/bash

set -e # Exit immediately if a command errors out

PROJECT_DIR=../../
TARGET_DIR="${PROJECT_DIR}/data/interim/repositories/"

mkdir -p "${TARGET_DIR}"

function extract_methods_from_repository {
  SOURCE_DIR=$1
  TARGET_FILE="${TARGET_DIR}/$(basename ${SOURCE_DIR}).csv"
  FILES=$(find "${SOURCE_DIR}" -name '*.java' -type f)
  FILE_COUNT=$(echo "${FILES}" | wc -l)

  echo "Extracting methods from ${FILE_COUNT} Java source files"

  echo "${FILES}" \
    | pv -l -s "${FILE_COUNT}" \
    | python "${PROJECT_DIR}/src/data/extract-java-methods.py" > "${TARGET_FILE}"
}

extract_methods_from_repository $1