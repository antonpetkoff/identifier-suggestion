package extractor.utils;

import java.io.File;

public class DirectoryCrawler {
  public interface FileHandler {
    void handle(int level, String path, File file);
  }

  public interface Filter {
    boolean isInterested(int level, String path, File file);
  }

  private FileHandler fileHandler;
  private Filter filter;

  public DirectoryCrawler(Filter filter, FileHandler fileHandler) {
    this.filter = filter;
    this.fileHandler = fileHandler;
  }

  public void explore(File root) {
    explore(0, "", root);
  }

  private void explore(int level, String path, File file) {
    if (file.isDirectory()) {
      for (File child : file.listFiles()) {
        // TODO: path.join?
        explore(level + 1, path + "/" + child.getName(), child);
      }
    } else {
      if (filter.isInterested(level, path, file)) {
        fileHandler.handle(level, path, file);
      }
    }
  }
}