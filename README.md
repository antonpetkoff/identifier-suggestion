# programming-tools

Programming tools, developed by learning from Big Code

## Setup

1. Have `pipenv` installed.

`export PIPENV_VENV_IN_PROJECT="enabled"`
<https://stackoverflow.com/questions/52540121/make-pipenv-create-the-virtualenv-in-the-same-folder>

## Scripts

### Java Method Name Extractor

1. `cd tools/java-extractor`
1. Build the jar: `./gradlew clean shadowJar`
1. Run the jar `time java -jar build/libs/java-extractor-all.jar ../../data/elasticsearch-master > output.csv`
