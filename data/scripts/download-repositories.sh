#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

REPOS_CSV=../meta/repositories.csv
TARGET_DIR=../external/repositories/

mkdir -p "${TARGET_DIR}"

echo "Reading repositories meta data from: ${REPOS_CSV}"

echo "Downloading and extracting repositories into: ${TARGET_DIR}"

function download_repo {
  OWNER_AND_REPO="$1"
  COMMIT_HASH="$2"

  wget -O - "https://github.com/${OWNER_AND_REPO}/archive/${COMMIT_HASH}.tar.gz" \
    | tar xfz - -C "${TARGET_DIR}"

  REPO="$(echo ${OWNER_AND_REPO} | cut -d '/' -f 2)"
  EXTRACTED_DIR_NAME="${REPO}-${COMMIT_HASH}"
  NEW_DIR_NAME="$(echo ${EXTRACTED_DIR_NAME} | sed -r 's/^(.*)-[^-]+$/\1/')"

  echo "Downloaded and extracted: ${EXTRACTED_DIR_NAME}"

  mv "${EXTRACTED_DIR_NAME}" "${NEW_DIR_NAME}"

  echo "Moved ${EXTRACTED_DIR_NAME} to ${NEW_DIR_NAME}"
}

tail -n +2 "${REPOS_CSV}" \
  | cut -d ';' -f -2 \
  | tr ';' ' ' \
  | while read REPO HASH; do download_repo "${REPO}" "${HASH}"; done
