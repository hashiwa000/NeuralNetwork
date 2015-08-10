package jp.hashiwa.nn;

/**
 * Created by Hashiwa on 2015/06/29.
 */
public abstract class NNLayerNode extends NNNode {
  private final NNNode[] inputs;
  private final double[] weights;
//  private Double cache = null;

  public NNLayerNode(int nodeNum) {
    this(new NNNode[nodeNum]);
  }
  public NNLayerNode(NNNode[] inputs) {
    this.inputs = new NNNode[inputs.length+1];
    this.inputs[0] = new NNInputNode(1);
    for (int i=0 ; i<inputs.length ; i++)
      this.inputs[i+1] = inputs[i];

    this.weights = new double[this.inputs.length];
    for (int i=0 ; i<this.weights.length ; i++)
      weights[i] = 2 * Math.random() - 1;
  }

  @Override
  public double getValue() {
//    if (cache != null) return cache;

    double v = 0;
    for (int i=0 ; i<inputs.length ; i++) {
      double w = weights[i];
      double x = inputs[i].getValue();
      v += w * x;
    }
    v = sigmoid(v);

    // zip
    // http://d.hatena.ne.jp/nowokay/20140321

//    cache = v;
    return v;
  }

  public int getInputNodeNum() {
    return inputs.length - 1;
  }

  public void setInputNode(int index, NNNode node) {
    // input of index==0 is fixed input "1"
    inputs[index + 1] = node;
//    clearCache();
  }

  public void setInputNodes(NNNode ... nodes) {
    for (int i=0 ; i<nodes.length ; i++)
      setInputNode(i, nodes[i]);
  }

//  public void clearCache() {
//    cache = null;
//  }

  protected double[] getWeights() {
    return weights;
  }

  protected NNNode[] getInputs() {
    return inputs;
  }

  protected double sigmoid(double x) {
    return 1 / (1 + Math.exp(-x));
  }
}
