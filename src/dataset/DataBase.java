package dataset;

import java.io.IOException;

public interface DataBase {
    void load(String imagePath, String lablePath, int number) throws IOException;
    double[] getData(int index);
    double[] getLabel(int index);
    int getSize();
    int getImageWidth();
    int getImageHeight();
}