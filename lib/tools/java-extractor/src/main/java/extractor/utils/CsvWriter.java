package extractor.utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class CsvWriter {
    public static void write(String pathName, String content) {
        try (PrintWriter writer = new PrintWriter(new File(pathName))) {
            writer.write(content);
            System.out.println("Wrote content to file!");
        } catch (FileNotFoundException e) {
            System.out.println(e.getMessage());
        }
    }
}
