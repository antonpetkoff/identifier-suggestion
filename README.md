# Programming tools, built by learning from Big Code

## Tasks

- [ ] Make everything reproducible
  - [ ] Write scripts for downloading the source code repositories (with wget)
  - [ ] Write scripts for generating the data-sets

## Setup

1. Have `pipenv` installed.

`export PIPENV_VENV_IN_PROJECT="enabled"`
<https://stackoverflow.com/questions/52540121/make-pipenv-create-the-virtualenv-in-the-same-folder>

## Scripts

### Java Method Name Extractor

1. `cd tools/java-extractor`
1. Build the jar: `./gradlew clean shadowJar`
1. Run the jar `java -jar build/libs/java-extractor-all.jar ../../data/repos/elasticsearch-master/ > ../../data/method-names/elastic-search.csv`

### Java Method Data Extractor 2.0

`find data/repos/elasticsearch-master/ -name '*.java' | pipenv run extract-java-methods | pv -l -s 110000 > data/method-names-rich/elasticsearch.csv`
