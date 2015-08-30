package jp.hashiwa.nn.graph;

import java.util.List;

/**
 * Created by Hashiwa on 2015/08/10.
 */
public interface LearningAlgorithm {
  void learn(List<double[]> data, List<double[]> expected);
}
