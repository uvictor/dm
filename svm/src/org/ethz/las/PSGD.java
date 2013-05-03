package org.ethz.las;

import org.apache.hadoop.conf.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

import java.io.*;
import java.util.*;

public class PSGD {
  // TODO: choose these three parameters
  final static int K = 64;
  final static double LAMBDA = 0.3;
  final static double LEARNING_RATE = 0.5;

  static Random randomGenerator = new Random();

  /**
   * The Map class has to make sure that the data is shuffled to the various machines.
   */
  public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, LongWritable, Text> {

    /**
     * Spread the data around on K different machines.
     */
    public void map(LongWritable key, Text value, OutputCollector<LongWritable, Text> output, Reporter reporter) throws IOException {
      LongWritable outputKey = new LongWritable(randomGenerator.nextInt(K));
      output.collect(outputKey, value);
    }
  }

  /**
   * Each of K reducers has to output one file containing the hyperplane.
   */
  public static class Reduce extends MapReduceBase implements Reducer<LongWritable, Text, NullWritable, Text> {

    /**
     * Construct a hyperplane given the subset of training examples.
     */
    public void reduce(LongWritable key, Iterator<Text> values, OutputCollector<NullWritable, Text> output, Reporter reporter) throws IOException {

      List<TrainingInstance> trainingSet = new LinkedList<TrainingInstance>();

      while (values.hasNext()) {
        String s = values.next().toString();
        TrainingInstance instance = new TrainingInstance(s);
        trainingSet.add(instance);
      }

      SVM model = new SVM(trainingSet, LAMBDA, LEARNING_RATE);

      /**
       * null is important here since we don't want to do additional preprocessing
       * to remove the key. The value should be the SVM model (take a look at method
       * toString in SVM.java.
       */
      Text outputValue = new Text();
      outputValue.set(model.toString());
      output.collect(null, outputValue);
    }
  }

  public static void main(String[] args) throws Exception {

    JobConf conf = new JobConf(PSGD.class);

    conf.setJobName("PSGD");

    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(Text.class);

    conf.setMapperClass(Map.class);
    conf.setReducerClass(Reduce.class);

    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);

    // set to the same K as above for optimal performance on the cluster
    // If you don't, you will likely have timeout problems.
    conf.setNumReduceTasks(PSGD.K);

    FileInputFormat.setInputPaths(conf, new Path(args[0]));
    FileOutputFormat.setOutputPath(conf, new Path(args[1]));

    JobClient.runJob(conf);
  }
}
