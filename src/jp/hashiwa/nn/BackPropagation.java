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
  public static final int DEFAULT_LEARNING_COUNT = 1000000;
  public static final double DEFAULT_MAX_DIFF = 0.1;
  private static final boolean DEBUG = true;

  private String logFileName;
  private Writer logWriter;
  private Graph graph;
  private int maxLearningCount;
  private double maxDiff;

  public BackPropagation(Graph g) {
    this(g, null, DEFAULT_LEARNING_COUNT, DEFAULT_MAX_DIFF);
  }

  public BackPropagation(Graph g, String logFileName) {
    this(g, logFileName, DEFAULT_LEARNING_COUNT, DEFAULT_MAX_DIFF);
  }

  public BackPropagation(Graph g, int maxLearningCount, double maxDiff) {
    this(g, null, maxLearningCount, maxDiff);
  }

  /**
   * Constructor.
   * @param g target Graph
   * @param logFileName log file name. If null, logging is disabled.
   * @param maxLearningCount max count of learning loop.
   * @param maxDiff max difference for terminating learning loop. If negative value, termination is disabled.
   */
  public BackPropagation(Graph g, String logFileName, int maxLearningCount, double maxDiff) {
    this.graph = g;
    this.logFileName = logFileName;
    this.maxLearningCount = maxLearningCount;
    this.maxDiff = maxDiff;
  }

  public void learn(List<double[]> data, List<double[]> expected ) {
    validate(data, expected);

    final int size = data.size();
    final int dim = data.get(0).length;
    boolean finished = false;
    double diff = -1;

    initLogger();

    for (int k=0 ; k<maxLearningCount && !finished ; k++) {
      IntStream.range(0, size).forEach(i ->
                      learnOne(data.get(i), expected.get(i))
      );

      diff = IntStream.range(0, size)
              .mapToDouble(i ->
                      IntStream.range(0, expected.get(i).length)
                      .mapToDouble(j -> Math.abs(graph.calculate(data.get(i))[j] - expected.get(i)[j]))
                      .sum()
              ).sum() / (size * dim);

      log(diff);

      if (0 <= maxDiff && diff < maxDiff) finished = true;
    }

    closeLogger();

    if (DEBUG) {
      if (0 <= maxDiff && !finished) {
        System.out.println("*** Learning is not complete. Average difference is " + diff + "(max is " + maxDiff + ").");
      }
      System.out.println("*** Learned graph is " + graph);
    }
  }

  private void validate(List<double[]> data, List<double[]> expected ) {
    if (data.size() != expected.size())
      new IllegalArgumentException("data length is invalid. " + data.size() + ", " + expected.size());

    for (int i=0 ; i<data.size() ; i++)
      if (data.get(i).length != expected.get(i).length)
        new IllegalArgumentException("data dimension at " + i + " is invalid. " + data.get(i).length + ", " + expected.get(i).length);
  }

  private void initLogger() {
    try {
      if (logFileName != null)
        logWriter = new BufferedWriter(new FileWriter(logFileName));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private void log(double diff) {
    if (logWriter == null) return;

    try {
      logWriter.write(diff + "\n");
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

  private void learnOne(double[] data, double[] expected) {
    final double K = 0.1;
    double[][] e;

    graph.setInputValues(data);
    e = getEs(expected);

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

  private double[][] getEs(double[] expected) {
    double[][] e = new double[graph.getHiddenNodeLayerSize()+1][];

    if (graph.getOutputNodeNum() != 1) {
      throw new RuntimeException("not support output node num : " + graph.getOutputNodeNum());
    }

    e[e.length-1] = new double[1];
    for (int i=0 ; i<expected.length ; i++)
      e[e.length-1][i] = e(i, expected[i]);

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
