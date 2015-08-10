package jp.hashiwa.nn;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Created by Hashiwa on 2015/07/12.
 */
public class Graph {
  private final NNNode[][] nodes;
  private final LearningAlgorithm alg = new BackPropagation(this);

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
    double[] ret = new double[getOutputNodeNum()];
    NNNode[] inputs = (NNNode[]) Arrays.stream(v).
                    mapToObj(NNInputNode::new).
                    toArray(value -> new NNNode[value]);

    setInputNodes(inputs);

    return IntStream.range(0, getOutputNodeNum()).
            mapToDouble(i -> getOutputNode(i).getValue()).
            toArray();
  }

  public void learn(List<double[]> data, List<Double> expected) {
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
  public NNHiddenLayerNode getHiddenNode(int jIndex, int iIndex) {
    if (getHiddenNodeLayerSize() < jIndex)
      throw new IndexOutOfBoundsException(Integer.toString(jIndex));

    return (NNHiddenLayerNode)nodes[jIndex][iIndex];
  }

  public int getOutputNodeNum() {
    return nodes[nodes.length-1].length;
  }
  public NNOutputLayerNode getOutputNode(int index) {
    return (NNOutputLayerNode)nodes[nodes.length-1][index];
  }

  public void setInputNodes(NNNode... newNodes) {
    for (NNNode target: nodes[0]) {
      ((NNLayerNode)target).setInputNodes(newNodes);
    }
  }

  private NNNode[] createNodes(int dimension, NNNode[] nextNodes) {
    NNNode[] nodes = new NNNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNHiddenLayerNode(nextNodes);

    return nodes;
  }

  private NNHiddenLayerNode[] createHiddenLayerNodes(int beforeDimension, int dimension) {
    NNHiddenLayerNode[] nodes = new NNHiddenLayerNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNHiddenLayerNode(beforeDimension);

    return nodes;
  }

  private NNOutputLayerNode[] createOutputLayerNodes(int dimension, NNNode[] nextNodes) {
    NNOutputLayerNode[] nodes = new NNOutputLayerNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNOutputLayerNode(nextNodes);

    return nodes;
  }

}
