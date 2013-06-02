#!/bin/bash

rm -rf bin/* Solution.jar

javac -cp ".:$HADOOP/hadoop-core-1.1.2.jar:$HADOOP/lib/*" src/org/ethz/las/*.java -d bin/

jar cvf PSGD.jar -C bin/ .
