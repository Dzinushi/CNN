package util;

import java.util.*;

public class Util {

    public static int max(final double[][] data){
        int max = Integer.MIN_VALUE;

        for (double[] aData : data) {
            for (double anAData : aData) {
                if (anAData > max) max = (int)anAData;
            }
        }
        
        return max;
    }

    public static int maxIndex(final double[] datas){
        double max = Double.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < datas.length; i++) {
            if (datas[i] > max){
                max = datas[i];
                index = i;
            }
        }

        return index;
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

    public static int[] randPerm(int size){
        Map<Integer, Boolean> uniqueIndex = new HashMap<>();
        int[] randpermData = new int[size];

        for (int i = 0; i < size; i++) {
            int index = (int) (Math.random() * size);

            if (!uniqueIndex.containsKey(index)){
                uniqueIndex.put(index, true);
                randpermData[i] = index;
            }
            else {
                boolean repeat = true;
                for (int j = 0; j < size & repeat; j++) {
                    index = (index + 1) % size;

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
}
