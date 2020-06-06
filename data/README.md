# Data

## Data Collection

### Searching for GitHub Repositories

GitHub exposes a REST API. One of the APIs is the [Search API](https://developer.github.com/v3/search/#search-repositories) for searching repositories.

The following command fetches the top 30 Java repositories, sorted by stargazer count
in descending order:

`curl 'https://api.github.com/search/repositories?q=language:java&sort=stars&order=desc&per_page=300' > top_java_repos_1.json`

The Search API is paginated and the returned Link header contains the link to the next page.
The next page can be obtained by the command:

`curl 'https://api.github.com/search/repositories?q=language%3Ajava&sort=stars&order=desc&per_page=300&page=2' > top_java_repos_2.json`

Then, this data can be reduced to the most important properties of the repositories like:

- number of commits
- number of stargazers
- number of contributors
- number of forks
- create time
- description

For this purpose, `jq` is very useful:

```shell
echo "full_name;stargazers_count;forks;created_at;clone_url;description" > repo_stats.csv

jq '.items[] | "\(.full_name);\(.stargazers_count);\(.forks);\(.created_at);\(.clone_url);\(.description)"' top_java_repos*.json | tr -d '"' >> repo_stats.csv
```
