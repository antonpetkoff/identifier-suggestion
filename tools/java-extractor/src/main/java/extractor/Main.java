package extractor;

import java.io.File;

public class Main {
  public static void main(String[] args) {
    File sourceCodePath = new File(
      "/home/tony/source/programming-tools/data/elasticsearch-master/"
    );

    ListClasses classNameCrawler = new ListClasses();
    classNameCrawler.listClasses(sourceCodePath);
  }
}
