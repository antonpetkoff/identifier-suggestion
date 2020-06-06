#!/bin/bash

function get_count {
  repo_owner_and_name="$1"
  property="$2"

  curl -I -k "https://api.github.com/repos/${repo_owner_and_name}/${property}?per_page=1" \
    | sed -n '/^[Ll]ink:/ s/.*"next".*page=\([0-9]*\).*"last".*/\1/p'
}

echo "full_name;stargazers_count;forks;commits;contributors;created_at;clone_url;description"

curl -k 'https://api.github.com/search/repositories?q=language:java&sort=stars&order=desc&per_page=300' \
  | jq '.items[] | "\(.full_name):\(.stargazers_count):\(.forks):\(.created_at):\(.clone_url):\(.description)"' \
  | tr -d '"' \
  | while IFS=':' read full_name stargazers_count forks created_at clone_url description; do
    commits=$(get_count "${full_name}" 'commits')
    contributors=$(get_count "${full_name}" 'contributors')

    echo "${full_name};${stargazers_count};${forks};${commits};${contributors};${created_at};${clone_url};${description}"
done
