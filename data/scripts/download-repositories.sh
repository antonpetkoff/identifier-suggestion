#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

REPOS_CSV=../meta/repositories.csv
DOWNLOAD_DIR=../external/repositories/
TARGET_DIR=../raw/repositories

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${TARGET_DIR}"

echo "Reading repositories meta data from: ${REPOS_CSV}"

echo "Downloading and extracting repositories into: ${DOWNLOAD_DIR}"

function download_repo {
  OWNER_AND_REPO="$1"
  COMMIT_HASH="$2"

  wget -O - "https://github.com/${OWNER_AND_REPO}/archive/${COMMIT_HASH}.tar.gz" \
    | tar xfz - -C "${DOWNLOAD_DIR}"

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

echo "Changing into target directory: ${TARGET_DIR}"
cd "${TARGET_DIR}"

echo "Picking only Java source files into ${TARGET_DIR}"

find . -type f -name '*.java' \
  | xargs -n 10000 cp --parents -t "${TARGET_DIR}" # execute cp with 10000 arguments at a time
