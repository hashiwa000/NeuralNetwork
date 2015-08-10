package jp.hashiwa.nn;

/**
 * Created by Hashiwa on 2015/07/02.
 */
public class NNHiddenLayerNode extends NNLayerNode {
  public NNHiddenLayerNode(int nodeNum) {
    super(nodeNum);
  }

  public NNHiddenLayerNode(NNNode[] inputs) {
    super(inputs);
  }
}
