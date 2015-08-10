package jp.hashiwa.nn;

import java.util.List;

/**
 * Created by Hashiwa on 2015/08/10.
 */
public interface LearningAlgorithm {
  void learn(List<double[]> data, List<Double> expected);
}
