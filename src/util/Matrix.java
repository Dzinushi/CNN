package util;

public class Matrix {
    private double[][] data;

    public Matrix(Size size){
        data = new double[size.x][size.y];
    }

    public void setValue(int indexRow, int indexCol, double value){
        data[indexRow][indexCol] = value;
    }

    public double getValue(int indexRow, int indexColumn){
        return data[indexRow][indexColumn];
    }

    public int getRowNum(){
        return data.length;
    }

    public int getColNum(){
        return data[0].length;
    }

    public Matrix rot180(){
        Matrix matrix = new Matrix(new Size(this.getRowNum(), this.getColNum()));
        int rowSize = matrix.getRowNum();
        int colSize = matrix.getColNum();

        for (int i = 0; i < rowSize / 2; i++) {
            for (int j = 0; j < colSize; j++) {
                double value = matrix.getValue(i,j);
                matrix.setValue(i, j, this.getValue(rowSize - i - 1, j));
                matrix.setValue(rowSize - i - 1, j, value);
            }
        }

        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize / 2; j++) {
                double value = matrix.getValue(i,j);
                matrix.setValue(i, j, this.getValue(i, colSize - j - 1));
                matrix.setValue(i, colSize - j - 1, value);
            }
        }

        return matrix;
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