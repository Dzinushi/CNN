package dataset;

import java.io.IOException;

/**
 * Для того, чтобы нейронная сеть смога корректно обучиться по базе данных реализующей этот интерфейс,
 * необходимо учесть некоторые условия:
 * 1) База данных должна содержать изображения одинакового размера.
 * 2) База данных должна быть нормализована от 0 до 1.
 */
public interface DataBase {
    void load(String imagePath, String lablePath, int number) throws IOException;
    double[] getData(int index);
    double[] getLabel(int index);
    int getSize();
    int getImageWidth();
    int getImageHeight();
}