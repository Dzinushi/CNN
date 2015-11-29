package dataset;

import util.Util;

import java.io.*;
import java.util.Arrays;
import java.util.Objects;


public class Mnist {
    private double[][] lable;
    private double[][] datas;

    private int size;
    private int imageWidth;
    private int imageHeight;

    /**
     *
     * @param imagePath - имя файла с данными о изображениях
     * @param lablePath - имя файла с данными значении на изображении
     * @param number    - число считываемых строк
     * @throws IOException
     */
    public void load(String imagePath, String lablePath, int number) throws IOException {

        DataInputStream imageFile = new DataInputStream(new FileInputStream(imagePath));
        DataInputStream lableFile = new DataInputStream(new FileInputStream(lablePath));

        int magicNumber = lableFile.readInt();

        if (magicNumber != 2049) {
            System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
            System.exit(0);
        }

        magicNumber = imageFile.readInt();

        if (magicNumber != 2051) {
            System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
            System.exit(0);
        }

        int numLabels = lableFile.readInt();
        int numImages = imageFile.readInt();
        int numRows = imageFile.readInt();
        int numCols = imageFile.readInt();

        lable = new double[number][10];
        datas = new double[number][numRows * numCols];

        imageWidth = numCols;
        imageHeight = numRows;

        if (numLabels != numImages) {
            System.err.println("Image file and label file do not contain the same number of entries.");
            System.err.println("  Label file contains: " + numLabels);
            System.err.println("  Image file contains: " + numImages);
            System.exit(0);
        }

        System.out.println("Start read Data Base ...");
        int imageForRead = number > 0 ? number : numImages;
        System.out.printf("All datas: %d\n", imageForRead);

        size = 0;
        for (int i = 0; i < imageForRead && lableFile.available() > 0; i++) {
            for (int rowIdx = 0, index = 0; rowIdx < numRows; rowIdx++) {
                for (int colIdx = 0; colIdx < numCols; colIdx++, index++) {
                    datas[i][index] = imageFile.readUnsignedByte();
                }
            }

            lable[i][lableFile.readByte()] = 1;
            size++;
        }

        System.out.println("End read data");

        normalizeData();
        //saveFormatData("train");
    }

    private void normalizeData(){
        System.out.println("Start normalize data ...");
        double max = Util.max(datas);
        datas = Util.normalize(datas, max);
        System.out.println("End normalize data");
    }

    public void saveFormatData(String filename){
        PrintWriter printWriter = null;

        try {
            printWriter = new PrintWriter(filename + ".format", "UTF-8");
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        assert printWriter != null;

        for (int i = 0; i < datas.length; i++) {
            for (int j = 0; j < datas[i].length; j++) {
                printWriter.print(datas[i][j] + ",");
            }
            int lable = 0;
            boolean repeat = true;
            for (int j = 0; j < this.lable[i].length & repeat; j++) {
                if (Objects.equals(this.lable[i][j], 1.0)){
                    lable = j;
                    repeat = false;
                }
            }

            printWriter.print(lable + "\n");
        }

        printWriter.close();
    }

    public double[] getData(int index){
        return datas[index];
    }

    public double[] getLable(int index){
        return lable[index];
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
}