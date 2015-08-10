package jp.hashiwa.nn;

/**
 * Created by Hashiwa on 2015/07/02.
 */
public class NNHiddenLayerNode extends NNLayerNode {
  public static final double K = 0.1;
  public static final double THRESHOLD = 0.0001;

  public NNHiddenLayerNode(int nodeNum) {
    super(nodeNum);
  }

  public NNHiddenLayerNode(NNNode[] inputs) {
    super(inputs);
  }
}
