package jp.hashiwa.nn;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Hashiwa on 2015/07/31.
 */
public class BackPropagation implements LearningAlgorithm {
  private Graph graph;

  public BackPropagation(Graph g) {
    this.graph = g;
  }

  public void learn(List<double[]> data, List<Double> expected ) {
    if (data.size() != expected.size())
      new IllegalArgumentException("data length is invalid. " + data.size() + ", " + expected.size());

    int size = data.size();
    for (int i=0 ; i<size ; i++)
      learnOne(data.get(i), expected.get(i));
  }

  public void learnOne(double[] data, double expected) {
    final double K = 0.1;
    double[][] outputs = getOutputs();
    double[][] e = getEs(expected);

    // outputs and e are known, so let's update weights.

    for (int i=0 ; i<graph.getHiddenNodeLayerSize() ; i++) {
      for (int j=0 ; j<graph.getHiddenNodeNum(i) ; j++) {
        NNLayerNode n = graph.getHiddenNode(i, j);
        double[] w = n.getWeights();
        for (int k=0 ; k<w.length ; k++) {
          w[k] += -1 * k * e[i][j] * outputs[i][j];
        }
      }
    }
  }

  private double[][] getEs(double expected) {
    double[][] e = new double[graph.getHiddenNodeLayerSize()+1][];

    if (graph.getOutputNodeNum() != 1) {
      throw new RuntimeException("not support output node num : " + graph.getOutputNodeNum());
    }

    e[e.length-1] = new double[1];
    e[e.length-1][0] = e(0, expected);

    for (int i=e.length-2 ; 0<=i ; i--) {
      e[i] = new double[graph.getHiddenNodeNum(i)];
      for (int j=0 ; j<e[i].length ; j++)
        e[i][j] = e(i, j, e[i+1]);
    }

    return e;
  }
  private double e(int outNodeIndex, double expected) {
    NNOutputLayerNode n = graph.getOutputNode(outNodeIndex);
    double actual = n.getValue();
    return (actual - expected) * actual * (1 - actual);
  }

  private double e(int hiddenLayerIndex, int hiddenNodeIndex, double[] e) {
    NNHiddenLayerNode node = graph.getHiddenNode(hiddenLayerIndex, hiddenNodeIndex);
    double actual = node.getValue();

    int nextLayerIndex = hiddenLayerIndex + 1;
    double sum = 0;
    for (int i=0 ; i<e.length ; i++) {
      NNLayerNode nextNode = nextLayerIndex < graph.getHiddenNodeLayerSize() ?
              graph.getHiddenNode(nextLayerIndex, i) :
              graph.getOutputNode(i);
      sum += e[i] * nextNode.getWeights()[hiddenNodeIndex];
    }

    return sum * actual * (1 - actual);
  }

  private double calc(double x, double y) {
    setInputs(x, y);
    return graph.getOutputNode(0).getValue();
  }

  private void setInputs(double x, double y) {
    NNNode value0 = new NNInputNode(x);
    NNNode value1 = new NNInputNode(y);

    graph.setInputNodes(value0, value1);
  }

  private double[][] getOutputs() {
    double[][] results = new double[graph.getHiddenNodeLayerSize()+1][];

    for (int i=0 ; i<graph.getHiddenNodeLayerSize() ; i++) {
      results[i] = new double[graph.getHiddenNodeNum(i)];
      for (int j=0 ; j<graph.getHiddenNodeNum(i) ; j++) {
        results[i][j] = graph.getHiddenNode(i, j).getValue();
      }
    }

    results[results.length-1] = new double[graph.getOutputNodeNum()];
    for (int k=0 ; k<graph.getOutputNodeNum() ; k++)
      results[results.length-1][k] = graph.getOutputNode(k).getValue();

    return results;
  }
}
