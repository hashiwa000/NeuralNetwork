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

  /**
   * learn using the expected value
   * @param y expected value
   * @return whether learning is finished or not
   */
  public boolean learnUsingExpectedValue(double y) {
    double u = getValue(); // actual value
    double dev = sigDeviation(u); // deviation of actual value
    double[] weights = getWeights();
    NNNode[] inputs = getInputs();

    double[] h = new double[inputs.length]; // ���ۂ̓��̓f�[�^
    double[] e = new double[weights.length]; // �C����
    double[] oldWeights = weights.clone();

    for (int i=0 ; i<weights.length ; i++)
      h[i] = inputs[i].getValue();

    for (int i=0 ; i<weights.length ; i++)
      e[i] = h[i] * (y - u) * dev;

    for (int i=0 ; i<e.length ; i++)
      weights[i] = weights[i] + K * e[i];

    boolean finished = true;
    for (int i=0 ; i<inputs.length ; i++) {
      if (inputs[i] instanceof NNHiddenLayerNode) {
        NNHiddenLayerNode n = (NNHiddenLayerNode) inputs[i];
        finished &= n.learnUsingFixedWeight(e[i], oldWeights[i]);
      }
    }

    for (int i=0 ; i<e.length ; i++)
      finished &= (Math.abs(e[i]) < THRESHOLD);

    return finished;
  }
}
