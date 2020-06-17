# Data

This directory contains data necessary for building the dataset
which is used for training the models.

Description of the contents of this directory:

```text
├── external       <- Data from third party sources.
├── interim        <- Intermediate data that has been transformed.
├── meta           <- Data about the data, e.g. the list of selected
│                     Java repositories and their properties.
├── processed      <- The final, canonical data sets for modeling.
├── raw            <- The original, immutable data dump.
└── scripts        <- Executable scripts for recreating the dataset.
```

## Dataset Reproduction

The dataset is can be reproduced in 3 steps described below.

The execution of all scripts must happen from this directory
and it is recommended to be done within an activated Python virtual environment.

### Step 1 - Data Collection

Download the Java source code repositories, described in `./meta/repositories.csv`
by executing `./scripts/download-repositories.sh`.

By default they are downloaded and extracted into `./external/repositories`
and all `.java` files are copied under `./raw/repositories`.

### Step 2 - Java Method Extraction

Execute `./scripts/extract-all-java-methods.sh` to extract all Java methods from the `.java` files from `./raw/repositories`.

The output is a set of CSV file for each repository with the set of extracted Java methods and their features.
These CSV files are stored under `./interim/repositories`.

### Step 3 - Data Preprocessing

The final data preprocessing is done by executing `./scripts/preprocess-data.sh`.
It executes a Python script which exposes a few parameters,
like the `tokenization_level` parameter which by default is `subtoken`.

By default, the preprocessed data is stored under `./processed/subtoken` where the last part of the path is the tokenization_level method.

The final processed data is in the HDF5 format and
has 3 Pandas data frames of training/validation/testing data
which can be used by a specific model.
Further data processing may be done by the model itself,
but the input data mustn't be modified - it should be treated as immutable.

## Additional Notes

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
