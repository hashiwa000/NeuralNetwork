package jp.hashiwa.nn;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Hashiwa on 2015/07/31.
 */
public class BackPropagation implements LearningAlgorithm {
  private String logFileName;
  private Writer logWriter;
  private Graph graph;

  public BackPropagation(Graph g) {
    this(g, null);
  }

  public BackPropagation(Graph g, String logFileName) {
    this.graph = g;
    this.logFileName = logFileName;
  }

  public void learn(List<double[]> data, List<Double> expected ) {
    if (data.size() != expected.size())
      new IllegalArgumentException("data length is invalid. " + data.size() + ", " + expected.size());

    int leanCnt = 100000;
    int size = data.size();

    initLogger();

    for (int k=0 ; k<leanCnt ; k++) {
      IntStream.range(0, size).forEach(i ->
                      learnOne(data.get(i), expected.get(i))
      );
      log(data, expected);
    }

    closeLogger();
  }

  private void initLogger() {
    try {
      if (logFileName != null)
        logWriter = new BufferedWriter(new FileWriter(logFileName));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private void log(List<double[]> data, List<Double> expected) {
    if (logWriter == null) return;

    double sum = 0;
    for (int i=0 ; i<data.size() ; i++) {
      double diff = graph.calculate(data.get(i))[0] - expected.get(i);
      sum += Math.abs(diff);
    }

    try {
      logWriter.write(sum + "\n");
    } catch(IOException ex) {
      ex.printStackTrace();
    }
  }

  private void closeLogger() {
    try {
      if (logWriter != null) {
        logWriter.flush();
        logWriter.close();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void learnOne(double[] data, double expected) {
    final double K = 0.1;
    double[][] e;

    graph.setInputValues(data);
    e = getEs(expected);

    // e are known, so let's update weights.

    for (int j=0 ; j<graph.getOutputNodeNum() ; j++) {
      NNOutputLayerNode n = graph.getOutputNode(j);
      double[] w = n.getWeights();
      for (int k=0 ; k<w.length ; k++) {
        double actual = n.getInputs()[k].getValue();
        w[k] += -1 * K * e[e.length-1][j] * actual;
      }
    }

    for (int i=graph.getHiddenNodeLayerSize()-1 ; 0<=i ; i--) {
      for (int j=0 ; j<graph.getHiddenNodeNum(i) ; j++) {
        NNLayerNode n = graph.getHiddenNode(i, j);
        double[] w = n.getWeights();
        for (int k=0 ; k<w.length ; k++) {
          double actual = n.getInputs()[k].getValue();
          w[k] += -1 * K * e[i][j] * actual;
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
}
