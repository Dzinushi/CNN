package util;

import java.util.Objects;

public class MatrixOperation {
    public static Matrix multiply(final Matrix matrix, final double value){
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                matrix.setValue(i, j, matrix.getValue(i,j) * value);
            }
        }

        return matrix;
    }

    public static Matrix divide(final Matrix matrix, final double value){
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                matrix.setValue(i, j, matrix.getValue(i, j) / value);
            }
        }

        return matrix;
    }

    public static Matrix rot180(final Matrix matrix){
        int rowSize = matrix.getRowNum();
        int colSize = matrix.getColNum();

        for (int i = 0; i < rowSize / 2; i++) {
            for (int j = 0; j < colSize; j++) {
                double value = matrix.getValue(i,j);
                matrix.setValue(i, j, matrix.getValue(rowSize - i - 1, j));
                matrix.setValue(rowSize - i - 1, j, value);
            }
        }

        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize / 2; j++) {
                double value = matrix.getValue(i,j);
                matrix.setValue(i, j, matrix.getValue(i, colSize - j - 1));
                matrix.setValue(i, colSize - j - 1, value);
            }
        }

        return matrix;
    }

    public static Matrix compression(Matrix map, Size size){
        int modWidth = Math.floorMod(map.getRowNum(), size.x);
        int modHeight = Math.floorMod(map.getColNum(), size.y);

        if (modHeight != 0 | modWidth != 0){
            System.out.printf("Ошибка. Ширина и высота карты не делятся нацело на [%d %d]\n", size.x, size.y);
            System.exit(-1);
        }

        int row = map.getRowNum() / size.x;
        int column = map.getColNum() / size.y;
        Matrix mapCompressed = new Matrix(new Size(row, column));

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {

                double max = Double.MIN_VALUE;

                for (int k = i * size.x; k < i * size.x + size.x; k++) {
                    for (int l = j * size.y; l < j * size.y + size.y; l++) {
                        if (map.getValue(k,l) > max)    max = map.getValue(i,j);
                    }
                }

                mapCompressed.setValue(i, j, max);
            }
        }

        return mapCompressed;
    }

    public static Matrix extend(Matrix map, Size size){
        int row = map.getRowNum();
        int column = map.getColNum();
        Matrix mapIncreased = new Matrix(new Size(row * size.x, column * size.y));

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {

                double max = Double.MIN_VALUE;

                for (int k = i * size.x; k < i * size.x + size.x; k++) {
                    for (int l = j * size.y; l < j * size.y + size.y; l++) {
                        mapIncreased.setValue(k, l, map.getValue(i,j));
                    }
                }
            }
        }

        return mapIncreased;
    }

    public static Matrix randomMapCNN(Size size){
        Matrix matrix = new Matrix(size);
        for (int i = 0; i < size.x; i++) {
            for (int j = 0; j < size.y; j++) {
                matrix.setValue(i, j, ((Math.random() * 2) - 0.05) / 10);
            }
        }
        return matrix;
    }

    public enum Op{
        SUM,
        MULTIPLY
    }

    public static Matrix operation(Matrix first, Matrix second, Op operation){
        if (!Objects.equals(first.getRowNum(), second.getRowNum()) | !Objects.equals(first.getColNum(), second.getColNum())){
            System.out.printf("Ошибка. Попытка сложения карт разного размера: [%d;%d] и [%d;%d]\n",
                    first.getRowNum(),
                    first.getColNum(),
                    second.getRowNum(),
                    second.getColNum());
            System.exit(-1);
        }

        Matrix sum = new Matrix(new Size(first.getRowNum(), first.getColNum()));

        for (int i = 0; i < first.getRowNum(); i++) {
            for (int j = 0; j < second.getColNum(); j++) {
                if (operation == Op.SUM){
                    double value = first.getValue(i,j) + second.getValue(i,j);
                    sum.setValue(i, j, value);
                }
                else if (operation == Op.MULTIPLY){
                    double value = (1 - first.getValue(i,j)) * second.getValue(i,j);
                    sum.setValue(i, j, value);
                }
            }
        }

        return sum;
    }

    public static double sumAllCells(Matrix matrix){
        double result = 0;
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                result += matrix.getValue(i,j);
            }
        }

        return result;
    }
}
