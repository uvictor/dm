#~/bin/bash

hadoop dfs -rmr /svm/output

hadoop jar Solution.jar org.ethz.las.PSGD /svm/input /svm/output
