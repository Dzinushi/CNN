package util;

import util.Size;

public class MapCNN {
    private double[][] data;

    public MapCNN(Size size){
        data = new double[size.x][size.y];
    }

    public void setValue(int indexRow, int indexCol, double value){
        data[indexRow][indexCol] = value;
    }

    public double getValue(int indexRow, int indexColumn){
        return data[indexRow][indexColumn];
    }
}
