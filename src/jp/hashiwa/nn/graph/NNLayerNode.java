package jp.hashiwa.nn.graph;

import java.util.stream.IntStream;

/**
 * Created by Hashiwa on 2015/06/29.
 */
public class NNLayerNode extends NNNode {
  private static final boolean PARALLEL = false;

  private final NNNode[] inputs;
  private final double[] weights;

  public NNLayerNode(int nodeNum) {
    this(new NNNode[nodeNum]);
  }

  public NNLayerNode(NNNode[] inputs) {
    this.inputs = new NNNode[inputs.length + 1];
    this.inputs[0] = new NNInputNode(1);
    for (int i = 0; i < inputs.length; i++)
      this.inputs[i + 1] = inputs[i];

    this.weights = new double[this.inputs.length];
    for (int i = 0; i < this.weights.length; i++)
      weights[i] = 2 * Math.random() - 1;
  }

  @Override
  public double getValue() {
    double v = 0;

    if (PARALLEL) {
      v = IntStream.range(0, inputs.length)
              .parallel()
              .mapToDouble(i -> inputs[i].getValue() * weights[i])
              .sum();
    } else {
      for (int i = 0; i < inputs.length; i++) {
        double w = weights[i];
        double x = inputs[i].getValue();
        v += w * x;
      }
    }

    v = sigmoid(v);

    return v;
  }

  public int getInputNodeNum() {
    return inputs.length - 1;
  }

  public void setInputNode(int index, NNNode node) {
    // input of index==0 is fixed input "1"
    inputs[index + 1] = node;
  }

  public void setInputNodes(NNNode... nodes) {
    for (int i = 0; i < nodes.length; i++)
      setInputNode(i, nodes[i]);
  }

  protected double[] getWeights() {
    return weights;
  }

  protected NNNode[] getInputs() {
    return inputs;
  }

  protected double sigmoid(double x) {
    return 1 / (1 + Math.exp(-x));
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append(getClass().getSimpleName()).append(" ");

    for (int i=0 ; i<inputs.length ; i++) {
      NNNode n = inputs[i];

      sb.append(i).append(": ");

      sb.append(weights[i]).append("=");

      if (n == null) sb.append("null");
      else sb.append("{").append(n.toString()).append("}");

      if (i != inputs.length-1)
        sb.append(", ");
    }

    return sb.toString();
  }
}
