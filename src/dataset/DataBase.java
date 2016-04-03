package dataset;

import java.io.IOException;

/**
 * Для того, чтобы нейронная сеть смога корректно обучиться по базе данных реализующей этот интерфейс,
 * необходимо учесть некоторые условия:
 * 1) База данных должна содержать изображения одинакового размера.
 * 2) База данных должна быть нормализована от 0 до 1.
 */
public interface DataBase {
    void load(String imagePath, String labelPath, int number) throws IOException;
    double[][] getData();
    void setLabel(int index, double[] label);
    void setData(int index, double[] data);
    void setSize(int size);
    void setImageWidth(int width);
    void setImageHeight(int imageHeight);
    double[] getData(int index);
    double[] getLabel(int index);
    int getSize();
    int getImageWidth();
    int getImageHeight();
}