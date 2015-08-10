package jp.hashiwa.nn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Hashiwa on 2015/06/29.
 */
public class Main {
  public static final int MAX_LEARN_LOOP = 1000000;
  Graph graph = new Graph(2, 2, 1);
  BackPropagation pb = new BackPropagation(graph);

  Main() {}

  void calcMain(double[][] data) {
    for (double[] d: data) {
      double actual = graph.calculate(d[0], d[1])[0];
      System.out.println(d[0] + "," + d[1] + "=" + actual);
    }
  }

  void learnMain(double[][] data) {
    for (double[] d: data) {
      double x = d[0];
      double y = d[1];
      double expected = d[2];
      double actual = graph.calculate(x, y)[0];
      System.out.println("[" + x + "," + y + "] expected=" + expected + ", actual=" + actual);
    }

    learn(data);

    for (double[] d: data) {
      double x = d[0];
      double y = d[1];
      double expected = d[2];
      double actual = graph.calculate(x, y)[0];
      System.out.println("[" + x + "," + y + "] expected=" + expected + ", actual=" + actual);
    }
  }

  void learn(double[][] dataAndExpected) {
//    boolean finished = false;
//    for (int i=0 ; i<MAX_LEARN_LOOP && !finished ; i++) {
//      finished = true;
//      for (double[] d: data) {
//        finished &= learnOne(d[0], d[1], d[2]);
//      }
//    }
//
//    System.out.println("finished = " + finished);

    List<double[]> data = Arrays.stream(dataAndExpected).
            map(d -> new double[] {d[0], d[1]}).
            collect(Collectors.toList());
    List<Double> expected = Arrays.stream(dataAndExpected).
            map(d -> new Double(d[0])).
            collect(Collectors.toList());

    graph.learn(data, expected);
  }

//  boolean learnOne(double x, double y, double expected) {
//    setInputs(x, y);
//    return graph.getOutputNode(0).learnUsingExpectedValue(expected);
//  }
//
//  double calculate(double x, double y) {
//    setInputs(x, y);
//    return graph.getOutputNode(0).getValue();
//  }
//
//  void setInputs(double x, double y) {
//    NNNode value0 = new NNInputNode(x);
//    NNNode value1 = new NNInputNode(y);
//
//    graph.setInputNodes(value0, value1);
//  }

  public static void main(String[] args) throws Exception {
    Main main = new Main();
    main.learnMain(readData("learn.txt", 3));
    main.calcMain(readData("data.txt", 2));
  }

  private static double[][] readData(String datafile, int elemNum) {
    List<List<Double>> data = new ArrayList<>();
    String line;
    String[] elems;

    try (BufferedReader br = new BufferedReader(new FileReader(datafile))) {
      while ((line=br.readLine()) != null) {

        List<Double> list = parseOneLine(line);
        if (list == null) continue;
        if (list.size() != elemNum) continue;

        data.add(list);
      }
    } catch (Exception e) {
      e.printStackTrace();
      return null;
    }

    System.out.println("=== " + data);

    return toArray(data);
  }

  private static List<Double> parseOneLine(String line) {
    List<Double> list = new ArrayList<>();
    String[] elems;

    line = line.trim();
    if (line.startsWith("#")) return null;
    if (line.equals("")) return null;

    elems = line.split(" ");

    Arrays.stream(elems).forEach(
            e -> list.add(Double.parseDouble(e)));

    return list;
  }

  private static double[][] toArray(List<List<Double>> list) {
    double[][] array = new double[list.size()][];

    for (int i=0 ; i<array.length ; i++) {
      List<Double> oneList = list.get(i);
      array[i] = new double[oneList.size()];
      for (int j=0 ; j<array[i].length ; j++)
        array[i][j] = oneList.get(j);
    }

    return array;
  }
}
