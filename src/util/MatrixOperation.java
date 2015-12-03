package util;

import net.Layer;

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

        Matrix matrixRot = new Matrix(new Size(rowSize, colSize));

        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize / 2; j++) {
                double value = matrix.getValue(i,j);
                matrixRot.setValue(i, j, matrix.getValue(i, colSize - j - 1));
                matrixRot.setValue(i, colSize - j - 1, value);
            }
        }

        for (int i = 0; i < rowSize / 2; i++) {
            for (int j = 0; j < colSize; j++) {
                double value = matrixRot.getValue(i,j);
                matrixRot.setValue(i, j, matrixRot.getValue(rowSize - i - 1, j));
                matrixRot.setValue(rowSize - i - 1, j, value);
            }
        }

        return matrixRot;
    }

    public static Matrix compression(final Matrix map, Size size){
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

//                double max = Double.MIN_VALUE;
//
//                for (int k = i * size.x; k < i * size.x + size.x; k++) {
//                    for (int l = j * size.y; l < j * size.y + size.y; l++) {
//                        if (map.getValue(k,l) > max)    max = map.getValue(i,j);
//                    }
//                }
//
//                mapCompressed.setValue(i, j, max);

                double sum = 0.0;
                for (int k = i * size.x; k < (i + 1) * size.x; k++) {
                    for (int l = j * size.y; l < (j + 1) * size.y; l++) {
                        sum += map.getValue(k,l);
                    }
                }

                mapCompressed.setValue(i, j, sum / (size.x * size.y));
            }
        }

        return mapCompressed;
    }

    public static Matrix extend(final Matrix map, Size size){
        int row = map.getRowNum();
        int column = map.getColNum();
        Matrix mapExtend = new Matrix(new Size(row * size.x, column * size.y));

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                for (int k = i * size.x; k < i * size.x + size.x; k++) {
                    for (int l = j * size.y; l < j * size.y + size.y; l++) {
                        mapExtend.setValue(k, l, map.getValue(i,j));
                    }
                }
            }
        }

        return mapExtend;
    }

    public static Matrix randomMapCNN(Size size){
        Matrix matrix = new Matrix(size);
        for (int i = 0; i < size.x; i++) {
            for (int j = 0; j < size.y; j++) {
                matrix.setValue(i, j, (/*(Math.random() * 2)*/Util.random.nextDouble() - 0.05) / 10);
            }
        }
        return matrix;
    }

    public enum Op{
        SUM,
        MULTIPLY
    }

    public static Matrix operation(final Matrix first, final Matrix second, Op operation){
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
                    double value = first.getValue(i,j) * second.getValue(i,j);
                    sum.setValue(i, j, value);
                }
            }
        }

        return sum;
    }

    public static double sumAllCells(final Matrix matrix){
        double result = 0;
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                result += matrix.getValue(i,j);
            }
        }

        return result;
    }

    public static Matrix clone(final Matrix matrix){
        Matrix cloneMatrix = new Matrix(new Size(matrix.getRowNum(), matrix.getColNum()));
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                cloneMatrix.setValue(i, j, matrix.getValue(i,j));
            }
        }

        return cloneMatrix;
    }

    // Запись в каждую ячейку значения "value - matrix[i][j]"
    public static Matrix valMinus(final Matrix matrix, double value){
        Matrix newMatrix = new Matrix(new Size(matrix.getRowNum(), matrix.getColNum()));
        for (int i = 0; i < newMatrix.getRowNum(); i++) {
            for (int j = 0; j < newMatrix.getColNum(); j++) {
                newMatrix.setValue(i, j, value - matrix.getValue(i, j));
            }
        }

        return newMatrix;
    }
}
