package jp.hashiwa.nn;

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
}
