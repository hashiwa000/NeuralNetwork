package jp.hashiwa.nn.graph;

/**
 * Created by Hashiwa on 2015/06/29.
 */
public class NNInputNode extends NNNode {
  private double value;

  public NNInputNode(double v) {
    this.value = v;
  }

  public double getValue() {
    return value;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append(getClass().getSimpleName()).append(" ");

    sb.append(value);

    return sb.toString();
  }
}
