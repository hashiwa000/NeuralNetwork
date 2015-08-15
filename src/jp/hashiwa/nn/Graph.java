package jp.hashiwa.nn;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Hashiwa on 2015/07/12.
 */
public class Graph {
  private final NNNode[][] nodes;
  private LearningAlgorithm alg;

  /**
   * dimension is
   * [# of input nodes][# of hidden nodes][[# of hidden nodes]...][# of output nodes]
   * @param dimension
   */
  public Graph(int... dimension) {
    if (dimension == null)
      throw new IllegalArgumentException("dimension is null.");

    if (dimension.length < 3)
      throw new IllegalArgumentException("the length of dimension is " + dimension.length);

    nodes = new NNNode[dimension.length-1][];

    // first hidden node setting
    int inputNodeNum = dimension[0];
    int firstHiddenNodeNum = dimension[1];
    nodes[0] = createHiddenLayerNodes(inputNodeNum, firstHiddenNodeNum);
    NNNode[] tmpNodes = nodes[0];

    // additional hidden node settings
    for (int i=1 ; i<dimension.length-2 ; i++) {
      tmpNodes = createNodes(dimension[i], tmpNodes);
      nodes[i] = tmpNodes;
    }

    // output node setting
    int lastDimension = dimension[dimension.length-1];
    nodes[nodes.length-1] = createOutputLayerNodes(lastDimension, tmpNodes);
  }

  public double[] calculate(double... v) {
    setInputValues(v);

    return IntStream.range(0, getOutputNodeNum()).
            mapToDouble(i -> getOutputNode(i).getValue()).
            toArray();
  }

  public void learn(LearningAlgorithm alg, List<double[]> data, List<Double> expected) {
    alg.learn(data, expected);
  }

  public int getHiddenNodeLayerSize() {
    return nodes.length - 1;
  }
  public int getHiddenNodeNum(int jIndex) {
    if (getHiddenNodeLayerSize() < jIndex)
      throw new IndexOutOfBoundsException(Integer.toString(jIndex));

    return nodes[jIndex].length;
  }
  public NNLayerNode getHiddenNode(int jIndex, int iIndex) {
    if (getHiddenNodeLayerSize() < jIndex)
      throw new IndexOutOfBoundsException(Integer.toString(jIndex));

    return (NNLayerNode)nodes[jIndex][iIndex];
  }

  public int getOutputNodeNum() {
    return nodes[nodes.length-1].length;
  }
  public NNLayerNode getOutputNode(int index) {
    return (NNLayerNode)nodes[nodes.length-1][index];
  }

  public void setInputValues(double... values) {
    NNNode[] inputs = Arrays.stream(values).
            mapToObj(NNInputNode::new).
            toArray(value -> new NNNode[value]);

    setInputNodes(inputs);
  }

  public void setInputNodes(NNNode... newNodes) {
    for (NNNode target: nodes[0]) {
      ((NNLayerNode)target).setInputNodes(newNodes);
    }
  }

  private NNNode[] createNodes(int dimension, NNNode[] nextNodes) {
    NNNode[] nodes = new NNNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNLayerNode(nextNodes);

    return nodes;
  }

  private NNLayerNode[] createHiddenLayerNodes(int beforeDimension, int dimension) {
    NNLayerNode[] nodes = new NNLayerNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNLayerNode(beforeDimension);

    return nodes;
  }

  private NNLayerNode[] createOutputLayerNodes(int dimension, NNNode[] nextNodes) {
    NNLayerNode[] nodes = new NNLayerNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNLayerNode(nextNodes);

    return nodes;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    NNNode[] outputNodes = nodes[nodes.length-1];

    for (int i=0 ; i<outputNodes.length ; i++) {
      NNNode n = outputNodes[i];

      sb.append(i).append(": ");

      if (n == null) sb.append("null");
      else sb.append("{").append(n.toString()).append("}");

      if (i != outputNodes.length-1)
        sb.append(", ");
    }

    return sb.toString();
  }
}
