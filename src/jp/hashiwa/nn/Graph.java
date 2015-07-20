package jp.hashiwa.nn;

/**
 * Created by Hashiwa on 2015/07/12.
 */
public class Graph {
  private final NNHiddenLayerNode[] inTermNodes;
  private final NNOutputLayerNode[] outTermNodes;

  public Graph(int... dimension) {
    if (dimension == null)
      throw new IllegalArgumentException("dimension is null.");

    if (dimension.length < 3)
      throw new IllegalArgumentException("the length of dimension is " + dimension.length);

    // first hidden node setting
    int inputNodeNum = dimension[0];
    int firstHiddenNodeNum = dimension[1];
    inTermNodes = createHiddenLayerNodes(inputNodeNum, firstHiddenNodeNum);
    NNNode[] nodes = inTermNodes;

    // additional hidden node settings
    for (int i=1 ; i<dimension.length-2 ; i++)
      nodes = createNodes(dimension[i], nodes);

    // output node setting
    int lastDimension = dimension[dimension.length-1];
    outTermNodes = createOutputLayerNodes(lastDimension, nodes);
  }

  public NNOutputLayerNode getOutputNode(int index) {
    return outTermNodes[index];
  }
  public void setInputNodes(NNNode... nodes) {
    for (NNLayerNode target: inTermNodes)
      target.setInputNodes(nodes);
  }

  private NNNode[] createNodes(int dimension, NNNode[] nextNodes) {
    NNNode[] nodes = new NNNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNHiddenLayerNode(nextNodes);

    return nodes;
  }

  private NNHiddenLayerNode[] createHiddenLayerNodes(int dimension, int nextDimension) {
    NNHiddenLayerNode[] nodes = new NNHiddenLayerNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNHiddenLayerNode(nextDimension);

    return nodes;
  }

  private NNOutputLayerNode[] createOutputLayerNodes(int dimension, NNNode[] nextNodes) {
    NNOutputLayerNode[] nodes = new NNOutputLayerNode[dimension];
    for (int i=0 ; i<nodes.length ; i++)
      nodes[i] = new NNOutputLayerNode(nextNodes);

    return nodes;
  }

}
