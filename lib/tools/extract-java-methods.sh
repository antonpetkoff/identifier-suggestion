#!/bin/bash

find data/repos/elasticsearch-master/ -name '*.java' | pipenv run extract-java-methods | pv -l -s 110000 > data/method-names-rich/elasticsearch.csv
