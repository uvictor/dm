package org.ethz.las;

import java.util.*;

public class SVM {

  // Hyperplane weights.
  RealVector weights;

  public SVM(RealVector weights) {
    this.weights = weights;
  }

  /**
   * Instantiates an SVM from a list of training instances, for a given
   * learning rate 'eta' and regularization parameter 'lambda'.
   */
  // TODO: Check this code after PSGD is implemented
  public SVM(List<TrainingInstance> trainingSet, double lambda, double eta) {
    weights = new RealVector(trainingSet.get(0).getFeatureCount());
    
    for (TrainingInstance ti : trainingSet) {
      double result = ti.getLabel() * weights.dotProduct(ti.getFeatures());

      // if violates margin
      if (result < 1) {
        // gradient step
        RealVector w = new RealVector(weights.getFeatures());
        w.add(ti.getFeatures().scaleThis(eta * ti.getLabel()));
        
        // reprojection
        weights = w.scaleThis(Math.min(1, 1 / (w.getNorm() * lambda)));
      }
    }
  }

  /**
   * Instantiates SVM from weights given as a string.
   */
  public SVM(String w) {
    List<Double> ll = new LinkedList<Double>();
    Scanner sc = new Scanner(w);
    sc.useLocale(Locale.US);
    while(sc.hasNext()) {
      double coef = sc.nextDouble();
      ll.add(coef);
    }
    sc.close();

    double[] weights = new double[ll.size()];
    int cnt = 0;
    for (Double coef : ll)
      weights[cnt++] = coef;

    this.weights = new RealVector(weights);
  }

  /**
   * Instantiates the SVM model as the average model of the input SVMs.
   */
  public SVM(List<SVM> svmList) {
    int dim = svmList.get(0).getWeights().getDimension();
    RealVector weights = new RealVector(dim);
    for (SVM svm : svmList)
      weights.add(svm.getWeights());

    this.weights = weights.scaleThis(1.0/svmList.size());
  }

  /**
   * Given a training instance it returns the result of sign(weights'instanceFeatures).
   */
  public int classify(TrainingInstance ti) {
    double result = ti.getFeatures().dotProduct(this.weights);
    if (result >= 0) return 1;
    else return -1;
  }

  public RealVector getWeights() {
    return this.weights;
  }

  @Override
  public String toString() {
    return weights.toString();
  }
}
