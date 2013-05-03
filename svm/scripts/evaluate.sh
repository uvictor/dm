#~/bin/bash

rm my.model

hadoop fs -getmerge /svm/output/ my.model

cd bin

java org.ethz.las.Performance ../my.model ../test.dat
