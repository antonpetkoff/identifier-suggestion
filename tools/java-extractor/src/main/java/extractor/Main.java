package extractor;

import java.io.File;

public class Main {
  public static void main(String[] args) {
    if (args.length != 1) {
      throw new IllegalArgumentException("Expecting a source directory");
    }

    File sourceCodePath = new File(args[0]);
    MethodNames.list(sourceCodePath);
  }
}
