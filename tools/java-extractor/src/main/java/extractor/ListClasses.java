package extractor;

import java.io.File;
import java.io.IOException;

import com.github.javaparser.JavaParser;
//import com.github.javaparser.ParseException;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import extractor.utils.DirectoryCrawler;

public class ListClasses {
  public static void listClasses(File projectDir) {
    new DirectoryCrawler(
      (level, path, file) -> path.endsWith(".java"),
      (level, path, file) -> {
        System.out.println(path);
        System.out.println("===============");

        try {
          new VoidVisitorAdapter<Object>() {
            @Override
            public void visit(ClassOrInterfaceDeclaration n, Object arg) {
              super.visit(n, arg);
              System.out.println(" * " + n.getName());
            }
          }.visit(JavaParser.parse(file), null);
          System.out.println(); // empty line
        } catch (IOException e) {
          new RuntimeException(e);
        }
      }
    ).explore(projectDir);
  }
}