package dataset;

import util.Util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Tennis implements DataBase {

    private double[][] data;
    private double[][] label;

    private int size;
    private int imageWidth;
    private int imageHeight;

    @Override
    public void load(String imagePath, String labelPath, int number) throws IOException {

        // считываем данные
        FileReader fileData = new FileReader(imagePath + ".txt");
        BufferedReader bData = new BufferedReader(fileData);

        String s;
        int i = 0;

        if ((s = bData.readLine()) != null){
            String[] arrayStrings = s.split(" ");
            size = Integer.valueOf(arrayStrings[0]);
            setImageHeight(Integer.valueOf(arrayStrings[1]));
        }

        data = new double[size][];

        while ((s = bData.readLine()) != null && i < size){
            String[] arrayStrings = s.split(" ");
            data[i] = Util.toDoubleArray(arrayStrings);
            i++;
        }
        if (data.length > 0){
            setImageWidth(data[0].length / imageHeight);
        }

        // считываем метки для данных
        FileReader fileLabel = new FileReader(labelPath + ".txt");
        BufferedReader bLabel = new BufferedReader(fileLabel);

        label = new double[size][];

        bLabel.readLine();
        i = 0;
        while ((s = bLabel.readLine()) != null && i < size){
            String[] arrayStrings = s.split(" ");
            label[i] = Util.toDoubleArray(arrayStrings);
            i++;
        }
    }

    @Override
    public double[][] getData() {
        return data;
    }

    @Override
    public void setLabel(int index, double[] label) {
        this.label[index] = label;
    }

    @Override
    public void setData(int index, double[] data) {
        this.data[index] = data;
    }

    @Override
    public void setSize(int size) {
        this.size = size;
        this.data = new double[size][];
        this.label = new double[size][];
    }

    @Override
    public void setImageWidth(int width) {
        this.imageWidth = width;
    }

    @Override
    public void setImageHeight(int imageHeight) {
        this.imageHeight = imageHeight;
    }

    @Override
    public double[] getData(int index) {
        return data[index];
    }

    @Override
    public double[] getLabel(int index) {
        return label[index];
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public int getImageWidth() {
        return imageWidth;
    }

    @Override
    public int getImageHeight() {
        return imageHeight;
    }
}
