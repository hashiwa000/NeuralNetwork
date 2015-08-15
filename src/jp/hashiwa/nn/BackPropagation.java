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
  public static final int DEFAULT_LEARNING_COUNT = 100000;

  private String logFileName;
  private Writer logWriter;
  private Graph graph;
  private int learningCount;

  public BackPropagation(Graph g) {
    this(g, null, DEFAULT_LEARNING_COUNT);
  }

  public BackPropagation(Graph g, String logFileName) {
    this(g, logFileName, DEFAULT_LEARNING_COUNT);
  }

  public BackPropagation(Graph g, int learningCount) {
    this(g, null, learningCount);
  }

  public BackPropagation(Graph g, String logFileName, int learningCount) {
    this.graph = g;
    this.logFileName = logFileName;
    this.learningCount = learningCount;
  }

  public void learn(List<double[]> data, List<Double> expected ) {
    if (data.size() != expected.size())
      new IllegalArgumentException("data length is invalid. " + data.size() + ", " + expected.size());

    int size = data.size();

    initLogger();

    for (int k=0 ; k<learningCount ; k++) {
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
    double[][] v;
    double[][] e;

    graph.setInputValues(data);
    v = getActualValues();
    e = getEs(expected);

    validateArraySize(v, e);

    // e are known, so let's update weights.

    for (int j=0 ; j<graph.getOutputNodeNum() ; j++) {
      NNLayerNode n = graph.getOutputNode(j);
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

  private void validateArraySize(double[][] a, double[][] b) {
    if (a.length != b.length) throw new Error("fatal : " + a.length + ", " + b.length);
    for (int i=0 ; i<a.length ; i++)
      if (a[i].length != b[i].length)
        throw new Error("fatal in " + i + " : " + a[i].length + ", " + b[i].length);
  }

  private double[][] getActualValues() {
    double[][] values = new double[graph.getHiddenNodeLayerSize()+1][];

    values[values.length-1] = new double[graph.getOutputNodeNum()];
    for (int j=0 ; j<values[values.length-1].length ; j++) {
      NNNode n = graph.getOutputNode(j);
      values[values.length - 1][j] = n.getValue();
    }

    for (int i=0 ; i<values.length-1 ; i++) {
      values[i] = new double[graph.getHiddenNodeNum(i)];
      for (int j=0 ; j<values[i].length ; j++) {
        NNNode n = graph.getHiddenNode(i, j);
        values[i][j] = n.getValue();
      }
    }

    return values;
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
    NNLayerNode n = graph.getOutputNode(outNodeIndex);
    double actual = n.getValue();
    return (actual - expected) * actual * (1 - actual);
  }

  private double e(int hiddenLayerIndex, int hiddenNodeIndex, double[] e) {
    NNLayerNode node = graph.getHiddenNode(hiddenLayerIndex, hiddenNodeIndex);
    double actual = node.getValue();
    int nextLayerIndex = hiddenLayerIndex + 1;
    int weightIndex = hiddenNodeIndex + 1;  // "+1" is for bias node

    double sum = 0;
    for (int i=0 ; i<e.length ; i++) {
      NNLayerNode nextNode = nextLayerIndex < graph.getHiddenNodeLayerSize() ?
              graph.getHiddenNode(nextLayerIndex, i) :
              graph.getOutputNode(i);
      sum += e[i] * nextNode.getWeights()[weightIndex];
    }

    return sum * actual * (1 - actual);
  }
}
