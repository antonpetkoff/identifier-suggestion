#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

REPOS_CSV=../meta/repositories.csv
TARGET_DIR=.

function download_repo {
  OWNER_AND_REPO="$1"
  COMMIT_HASH="$2"

  wget -O - "https://github.com/${OWNER_AND_REPO}/archive/${COMMIT_HASH}.tar.gz" \
    | tar xfz - -C "${TARGET_DIR}"
}

tail -n +2 "${REPOS_CSV}" \
  | cut -d ';' -f -2 \
  | tr ';' ' ' \
  | while read REPO HASH; do download_repo "${REPO}" "${HASH}"; done
