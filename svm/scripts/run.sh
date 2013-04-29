#~/bin/bash

hadoop dfs -rmr /svm/output

hadoop jar PSGD.jar org.ethz.las.PSGD /svm/input /svm/output
