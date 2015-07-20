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

  /**
   * learn using the expected value
   * @param e modified value
   * @param h
   * @return whether learning is finished or not
   */
  public boolean learnUsingFixedWeight(double e, double h) {
    double[] weights = getWeights();
    NNNode[] inputs = getInputs();
    double[] oldWeights = weights.clone();
    double g = getValue();

    double tmp = e * h * sigDeviation(g);

    boolean finished = true;
    for (int i=0 ; i<weights.length ; i++) {
      double x = inputs[i].getValue();
      double diff = x * tmp;
      weights[i] += K * diff;

      finished &= (Math.abs(diff) < THRESHOLD);
    }

    for (int i=0 ; i<inputs.length ; i++) {
      if (inputs[i] instanceof NNHiddenLayerNode) {
        NNHiddenLayerNode n = (NNHiddenLayerNode)inputs[i];
        finished &= n.learnUsingFixedWeight(e, oldWeights[i]);
      }
    }

    return finished;
  }
}
