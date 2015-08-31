package jp.hashiwa.nn.graph;

import java.util.Arrays;

/**
 * Created by Hashiwa on 2015/08/31.
 */
public class NNCachableLayerNode extends NNLayerNode {
  private boolean cached = false;
  private double cachedValue = Double.MAX_VALUE;

  public NNCachableLayerNode(int nodeNum) {
    super(nodeNum);
  }

  public NNCachableLayerNode(NNNode[] inputs) {
    super(inputs);
  }

  @Override
  public double getValue() {
    if (!cached) {
      cachedValue = super.getValue();
      cached = true;
    }

    return cachedValue;
  }

  public void clearFollowingNodesCache() {
    if (!cached) return;

    cached = false;

    NNNode[] nodes = getInputs();
    if (nodes != null) {
      Arrays.stream(nodes)
              .filter(n -> n instanceof NNCachableLayerNode)
              .forEach(n -> ((NNCachableLayerNode)n).clearFollowingNodesCache());
    }
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append("Cache(").append(cached ? "cached" : "not cached").append("):");
    buf.append(super.toString());

    return buf.toString();
  }
}
