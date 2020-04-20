package extractor;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ParseProblemException;

import extractor.utils.DirectoryCrawler;

public class MethodNames {
  public static final char DELIMITER = ';';

  public static void list(File projectDir) {
    PrintWriter printer = new PrintWriter(System.out);

    printer
      .append("file")
      .append(DELIMITER)
      .append("id")
      .append(DELIMITER)
      .append("type")
      .append('\n');

    new DirectoryCrawler(
      (level, path, file) -> path.endsWith(".java"),
      (level, path, file) -> {
        try {
          new VoidVisitorAdapter<>() {
            @Override
            public void visit(MethodDeclaration node, Object arg) {
              super.visit(node, arg);

              printer
                .append(path)
                .append(DELIMITER)
                .append(node.getName().getIdentifier())
                .append(DELIMITER)
                .append(node.getType().toString())
                .append('\n');
            }
          }.visit(JavaParser.parse(file), null);
        } catch (IOException | ParseProblemException e) {
          // swallow exception
//          System.err.println(e.getMessage());
        }
      }
    ).explore(projectDir);

    printer.flush();
  }
}