package jp.hashiwa.nn.view;

import jp.hashiwa.nn.BackPropagation;
import jp.hashiwa.nn.Graph;
import jp.hashiwa.nn.LearningAlgorithm;
import jp.hashiwa.nn.Main;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.text.DecimalFormat;

/**
 * Created by Hashiwa on 2015/08/11.
 */
public class Validation extends JFrame implements Runnable {
  private Main main;
  private int width = 100;
  private int height = 100;

  Validation() {
    Graph graph = new Graph(2, 2, 1);
    LearningAlgorithm alg = new BackPropagation(graph, 10000);
    main = new Main(graph, alg);

    setLayout(new FlowLayout());
    setSize(500, 500);
    setVisible(true);
    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    new Thread(this).start();
  }

  @Override
  public void run() {
    Container c = getContentPane();
    c.add(new Canvas());

    while (true) {
      try {
        System.in.read();
      } catch(IOException e) {}

      main.learnMain(main.readData("learn.txt", 3));
      c.add(new Canvas());
      repaint();
    }
  }

  public static void main(String[] args) throws Exception {
    new Validation();
  }

  private class Canvas extends JPanel {
    private final int MAX = 10;
    private final DecimalFormat formatter = new DecimalFormat("00.00");
    private double values[][];

    public Canvas() {
      Graph graph = main.getGraph();
      System.out.println(graph);

      values = new double[width][];
      for (int i=0 ; i<width ; i++) {
        values[i] = new double[height];
        for (int j=0 ; j<height ; j++) {
          double x = (double)i / width * MAX;
          double y = (double)j / height * MAX;

          graph.setInputValues(x, y);
          double v = graph.getOutputNode(0).getValue();

          values[i][j] = v;
        }
      }
      System.out.println("-----------------");

      setPreferredSize(new Dimension(100, 100));
    }

    @Override
    public void paint(Graphics g) {
      g.setColor(Color.white);
      g.fillRect(0, 0, width, height);

      for (int i=0 ; i<width ; i++) {
        for (int j=0 ; j<height ; j++) {
          double v = values[i][j];
          Color c = getColor(v);

          g.setColor(c);
          g.fillRect(i, height - j, 1, 1);
        }
      }

//      double[][] data = {
//              { 2, 3 },
//              { 5, 2 },
//              { 5, 9 },
//              { 9, 2 },
//      };
//
//      g.setColor(Color.black);
//      for (double[] p: data)
//        g.fillRect((int)(p[0] / MAX * width),(int)(height - (p[1] / MAX * height)), 1, 1);
    }

    private Color getColor(double v) {
      if (v < 0)   return Color.yellow;
      if (v < 0.5) return new Color(255-(int)(255*v), 0, 0);
      if (v < 1.0) return new Color(0, 0, (int)(255*v));
      return Color.green;
    }
  }
}
