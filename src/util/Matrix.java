package util;

public class Matrix {
    private double[][] data;

    public Matrix(Size size){
        data = new double[size.x][size.y];
    }

    public void setValue(final int indexRow, final int indexCol, final double value){
        data[indexRow][indexCol] = value;
    }

    public double getValue(final int indexRow,final int indexColumn){
        return data[indexRow][indexColumn];
    }

    public int getRowNum(){
        return data.length;
    }

    public int getColNum(){
        return data[0].length;
    }

    public String toString(){
        StringBuilder stringBuilder = new StringBuilder();
        for (double[] aData : data) {
            for (double anAData : aData) {
                stringBuilder.append(anAData).append(" ");
            }
            stringBuilder.append("\n");
        }
        return stringBuilder.toString();
    }
}