#!/bin/bash

cat $1 | egrep '.+;.+;.+' | egrep -v '\/\*' | egrep -v '//' | egrep -v 'NULL' > $2

