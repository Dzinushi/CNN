package dataset;

import util.Util;
import java.io.*;


public class Mnist implements DataBase{
    private double[][] label;
    private double[][] data;

    private int size;
    private int imageWidth;
    private int imageHeight;
    private int max;

    /**
     *
     * @param imagePath - имя файла с данными о изображениях
     * @param labelPath - имя файла с данными значении на изображении
     * @param number    - число считываемых строк
     * @throws IOException
     */
    public void load(String imagePath, String labelPath, int number) throws IOException {

        DataInputStream imageFile = new DataInputStream(new FileInputStream(imagePath));
        DataInputStream labelFile = new DataInputStream(new FileInputStream(labelPath));

        int magicNumber = labelFile.readInt();

        if (magicNumber != 2049) {
            //System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
            System.exit(0);
        }

        magicNumber = imageFile.readInt();

        if (magicNumber != 2051) {
            //System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
            System.exit(0);
        }

        int numLabels = labelFile.readInt();
        int numImages = imageFile.readInt();
        int numRows = imageFile.readInt();
        int numCols = imageFile.readInt();

        label = new double[number][10];
        data = new double[number][numRows * numCols];

        imageWidth = numCols;
        imageHeight = numRows;

        if (numLabels != numImages) {
            System.err.println("Image file and label file do not contain the same number of entries.");
            System.err.println("  Label file contains: " + numLabels);
            System.err.println("  Image file contains: " + numImages);
            System.exit(0);
        }

        //System.out.println("Start read Data Base ...");
        int imageForRead = number > numImages ? numImages : number;
        //System.out.printf("All data: %d\n", imageForRead);

        size = 0;
        for (int i = 0; i < imageForRead && labelFile.available() > 0; i++) {
            for (int rowIdx = 0, index = 0; rowIdx < numRows; rowIdx++) {
                for (int colIdx = 0; colIdx < numCols; colIdx++, index++) {
                    data[i][index] = imageFile.readUnsignedByte();
                }
            }

            label[i][labelFile.readByte()] = 1;
            size++;
        }

        //System.out.println("End read data");

        normalizeData();
    }

    private void normalizeData(){
        //System.out.println("Start normalize data ...");
        max = Util.max(data);
        data = Util.normalize(data, max);
        //System.out.println("End normalize data");
    }

    public double[] getData(int index){
        return data[index];
    }

    public double[] getLabel(int index){
        return label[index];
    }

    public int getSize(){
        return size;
    }

    public int getImageWidth(){
        return imageWidth;
    }

    public int getImageHeight(){
        return imageHeight;
    }

    @Override
    public int getMaxValue() {
        return max;
    }
}