package util;

import java.util.*;

public class Util {

    public static double max(final double[][] data){
        double max = Integer.MIN_VALUE;

        for (double[] aData : data) {
            for (double anAData : aData) {
                if (anAData > max) max = anAData;
            }
        }

        return max;
    }

    public static double[][] normalize(double[][] data, double max){
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                if (!Objects.equals(data[i][j],0)){
                    data[i][j] /= max;
                }
            }
        }
        return data;
    }

    public static int[] randPerm(int number){
        Map<Integer, Boolean> uniqueIndex = new HashMap<>();
        int[] randpermData = new int[number];

        for (int i = 0; i < number; i++) {
            int index = (int) (Math.random() * number);

            if (!uniqueIndex.containsKey(index)){
                uniqueIndex.put(index, true);
                randpermData[i] = index;
            }
            else {
                boolean repeat = true;
                for (int j = 0; j < number & repeat; j++) {
                    index = (index + 1) % number;

                    if (!uniqueIndex.containsKey(index)) {
                        uniqueIndex.put(index, true);
                        randpermData[i] = index;
                        repeat = false;
                    }
                }
            }
        }

        return randpermData;
    }

    public static MapCNN randomMapCNN(Size size){
        MapCNN mapCNN = new MapCNN(size);
        for (int i = 0; i < size.x; i++) {
            for (int j = 0; j < size.y; j++) {
                mapCNN.setValue(i,j, ((Math.random() * 2) - 0.05) / 10);
            }
        }
        return mapCNN;
    }

    public static MapCNN compression(MapCNN map, Size size){
        int modWidth = Math.floorMod(map.getRowNum(), size.x);
        int modHeight = Math.floorMod(map.getColNum(), size.y);

        if (modHeight != 0 | modWidth != 0){
            System.out.printf("Ошибка. Ширина и высота карты не делятся нацело на [%d %d]\n", size.x, size.y);
            System.exit(-1);
        }

        int row = map.getRowNum() / size.x;
        int column = map.getColNum() / size.y;
        MapCNN mapCompressed = new MapCNN(new Size(row, column));

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
}
