package jp.hashiwa.nn.view;

import jp.hashiwa.nn.Graph;
import jp.hashiwa.nn.Main;

import javax.swing.*;
import java.awt.*;

/**
 * Created by Hashiwa on 2015/08/11.
 */
public class Validation extends JFrame {
  private Graph graph;
  private int width = 100;
  private int height = 100;

  Validation() {
    Main main = new Main();
//    main.learnMain(main.readData("learn.txt", 3));
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
    @Override
    public void paint(Graphics g) {
      g.setColor(Color.white);
      g.fillRect(0, 0, width, height);

      for (int i=0 ; i<10 ; i++) {
        for (int j=0 ; j<10 ; j++) {
          double x = (double)i / 10;
          double y = (double)j / 10;

          Color c;
          graph.setInputValues(x, y);
          double v = graph.getOutputNode(0).getValue();

//          double rnd = Math.random();
//          if (rnd < 0.25) v = -1;
//          else if (rnd < 0.5) v = 0.2;
//          else if (rnd < 0.75) v = 0.7;
//          else v = 1.2;


          if (v < 0) c = Color.yellow;
          else if (v < 0.5) c = Color.red;
          else if (v < 1.0) c = Color.blue;
          else c = Color.green;

          System.out.println(i + ", " + j + " = " + v);

          g.setColor(c);
          g.fillRect(i * 10, j * 10, 10, 10);

        }
      }
    }
  }
}
