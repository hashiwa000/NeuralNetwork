package jp.hashiwa.nn.graph;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Hashiwa on 2015/07/12.
 */
public class Graph {
  public static boolean NOT_USE_CACHED_NODE = Boolean.getBoolean("jp.hashiwa.nn.graph.notUseCachedNode");
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

  public void learn(LearningAlgorithm alg, List<double[]> data, List<double[]> expected) {
    validate(data, expected);

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
    for (int i=0 ; i<getOutputNodeNum() ; i++) {
      NNLayerNode n = getOutputNode(i);
      if (n instanceof NNCachableLayerNode)
        ((NNCachableLayerNode)n).clearFollowingNodesCache();
    }

    for (NNNode target: nodes[0]) {
      ((NNLayerNode)target).setInputNodes(newNodes);
    }
  }

  private NNNode[] createNodes(int dimension, NNNode[] nextNodes) {
    NNNode[] nodes = new NNNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = NOT_USE_CACHED_NODE ?
              new NNLayerNode(nextNodes) :
              new NNCachableLayerNode(nextNodes);

    return nodes;
  }

  private NNLayerNode[] createHiddenLayerNodes(int beforeDimension, int dimension) {
    NNLayerNode[] nodes = new NNLayerNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = NOT_USE_CACHED_NODE ?
              new NNLayerNode(beforeDimension) :
              new NNCachableLayerNode(beforeDimension);

    return nodes;
  }

  private NNLayerNode[] createOutputLayerNodes(int dimension, NNNode[] nextNodes) {
    NNLayerNode[] nodes = new NNLayerNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = NOT_USE_CACHED_NODE ?
              new NNLayerNode(nextNodes) :
              new NNCachableLayerNode(nextNodes);

    return nodes;
  }

  private void validate(List<double[]> data, List<double[]> expected ) {
    if (data.size() != expected.size())
      new IllegalArgumentException("data length is invalid. " + data.size() + ", " + expected.size());

    for (int i=0 ; i<data.size() ; i++)
      if (data.get(i).length != expected.get(i).length)
        new IllegalArgumentException("data dimension at " + i + " is invalid. " + data.get(i).length + ", " + expected.get(i).length);

    if (data.size() != 0) {
      int dataLen = data.get(0).length;
      int nodeLen = getHiddenNodeNum(0);
      if (nodeLen != dataLen)
        throw new IllegalArgumentException("Illegal data length : expected=" + nodeLen + ", actual=" + dataLen);
    }

    if (expected.size() != 0) {
      int expectedLen = expected.get(0).length;
      int nodeLen = getOutputNodeNum();
      if (nodeLen != expectedLen)
        throw new IllegalArgumentException("Illegal expected data length : expected=" + nodeLen + ", actual=" + expectedLen);
    }
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
