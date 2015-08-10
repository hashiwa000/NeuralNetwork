package jp.hashiwa.nn;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Hashiwa on 2015/07/02.
 */
public class NNOutputLayerNode extends NNLayerNode {
  public static final double K = 0.1;
  public static final double THRESHOLD = 0.0001;

  public NNOutputLayerNode(int nodeNum) {
    super(nodeNum);
  }

  public NNOutputLayerNode(NNNode[] inputs) {
    super(inputs);
  }
}
