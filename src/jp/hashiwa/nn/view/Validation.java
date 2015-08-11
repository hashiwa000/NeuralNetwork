package jp.hashiwa.nn.view;

import jp.hashiwa.nn.Graph;
import jp.hashiwa.nn.Main;

import javax.swing.*;
import java.awt.*;
import java.text.DecimalFormat;

/**
 * Created by Hashiwa on 2015/08/11.
 */
public class Validation extends JFrame {
  private Graph graph;
  private int width = 100;
  private int height = 100;

  Validation() {
    Main main = new Main();
    main.learnMain(main.readData("learn.txt", 3));
    graph = main.getGraph();

    Container c = getContentPane();
    c.add(new Canvas(), BorderLayout.CENTER);

    setSize(150, 150);
    setVisible(true);
    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
  }

  public static void main(String[] args) throws Exception {
    new Validation();
  }

  private class Canvas extends JPanel {
    private final int MAX = 10;
    private final DecimalFormat formatter = new DecimalFormat("00.00");

    @Override
    public void paint(Graphics g) {
      g.setColor(Color.white);
      g.fillRect(0, 0, width, height);

      for (int i=0 ; i<width ; i++) {
        for (int j=0 ; j<height ; j++) {
          double x = (double)i / width * MAX;
          double y = (double)j / height * MAX;

          graph.setInputValues(x, y);
          double v = graph.getOutputNode(0).getValue();
          Color c = getColor(v);

//          double rnd = Math.random();
//          if (rnd < 0.25) v = -1;
//          else if (rnd < 0.5) v = 0.2;
//          else if (rnd < 0.75) v = 0.7;
//          else v = 1.2;

          System.out.println(
                  formatter.format(x) + ", " +
                  formatter.format(y) + " = " +
                  formatter.format(v));

          g.setColor(c);
          g.fillRect(i, height - j, 1, 1);
        }
      }
    }

    private Color getColor(double v) {
      if (v < 0)   return Color.yellow;
      if (v < 0.5) return Color.red;
      if (v < 1.0) return Color.blue;
      return Color.green;
    }
  }
}
