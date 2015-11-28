package net;

import util.Matrix;
import util.MatrixOperation;
import util.Size;
import java.util.ArrayList;
import java.util.List;

public class Layer {
    public LayerType type;
    private int mapOutNumber;
    private Size mapsSize;
    private Size kernelSize;
    private Size compressSise;
    private int classNum = -1;

    private List<List<Matrix>> kernel;  // ядра свертки
    private List<List<Matrix>> error;   // ошибки карт
    private List<List<Matrix>> mapOut;  // набор карт
    private double[] t;                 // пороговые значения

    enum LayerType {
        INPUT, OUTPUT, CONVOLUTION, SUBSAMPLING
    }

    // Создание входного слоя
    public static Layer inputLayer(Size mapSize) {
        Layer layer = new Layer();
        layer.type = LayerType.INPUT;
        layer.mapOutNumber = 1;
        layer.setMapSize(mapSize);
        return layer;
    }

    // Создание сверточного слоя
    public static Layer convLayer(int outMapNum, Size kernelSize) {
        Layer layer = new Layer();
        layer.type = LayerType.CONVOLUTION;
        layer.mapOutNumber = outMapNum;
        layer.kernelSize = kernelSize;
        return layer;
    }

    // Создание субдескритизирующего слоя
    public static Layer sampLayer(Size scaleSize) {
        Layer layer = new Layer();
        layer.type = LayerType.SUBSAMPLING;
        layer.compressSise = scaleSize;
        return layer;
    }

    // Создание выходного слоя
    public static Layer outputLayer(int classNum) {
        Layer layer = new Layer();
        layer.classNum = classNum;
        layer.type = LayerType.OUTPUT;
        layer.mapsSize = new Size(1, 1);
        layer.mapOutNumber = classNum;
        return layer;
    }

    public void setMapSize(Size mapSize) {
        this.mapsSize = mapSize;
    }

    // Задаем размерность ядра свертки
    public void setKernelSize(int mapOutNumber){
        kernel = new ArrayList<>();

        for (int i = 0; i < mapOutNumber; i++) {
            List<Matrix> matrixes = new ArrayList<>();
            for (int j = 0; j < getMapOutNumber(); j++) {
                Matrix matrix = new Matrix(getKernelSize());
                matrixes.add(matrix);
            }
            kernel.add(matrixes);
        }

        setKernelRandomValue();
    }

    public void setKernelMatrix(final int mapOutIndex,final int index, final Matrix matrix){
        kernel.get(mapOutIndex).set(index, matrix);
    }

    public void setOutKernel(int mapOutNumber, Size kernelSize){
        this.kernelSize = kernelSize;
        kernel = new ArrayList<>();

        for (int i = 0; i < mapOutNumber; i++) {
            List<Matrix> matrixes = new ArrayList<>();
            for (int j = 0; j < getMapOutNumber(); j++) {
                Matrix matrix = new Matrix(getKernelSize());
                matrixes.add(matrix);
            }
            kernel.add(matrixes);
        }

        setKernelRandomValue();
    }

    private void setKernelRandomValue(){
        for (List<Matrix> aKernel : kernel) {
            for (int j = 0; j < getMapOutNumber(); j++) {
                aKernel.set(j, MatrixOperation.randomMapCNN(kernelSize));
            }
        }
    }

    // Задание размерности порога
    public void setTSize(){
        t = new double[getMapOutNumber()];
    }

    // Задать значение порога по индексу
    public void setTValue(int outMapNumber, double value){
        t[outMapNumber] = value;
    }

    // Задаем размерность ошибок
    public void setErrorSize(int batchsize){
        error = new ArrayList<>(batchsize);
        for (int i = 0; i < batchsize; i++) {
            List<Matrix> matrixes = new ArrayList<>();
            for (int j = 0; j < getMapOutNumber(); j++) {
                Matrix matrix = new Matrix(getMapsSize());
                matrixes.add(matrix);
            }
            error.add(matrixes);
        }
    }

    public void setErrorValue(int indexMapOut, int index, int i, int j, double value){
        List<Matrix> listMap = error.get(indexMapOut);
        Matrix map  = listMap.get(index);
        map.setValue(i, j, value);
    }

    public void setErrorMap(int indexMapOut, int index, Matrix map){
        error.get(indexMapOut).set(index, map);
    }

    // Задаем размерность карт на выходе
    public void setMapOutSize(int batchsize){
        mapOut = new ArrayList<>(batchsize);
        for (int i = 0; i < batchsize; i++) {
            List<Matrix> matrixes = new ArrayList<>();
            for (int j = 0; j < getMapOutNumber(); j++) {
                Matrix matrix = new Matrix(getMapsSize());
                matrixes.add(matrix);
            }
            mapOut.add(matrixes);
        }
    }

    public void setMapOutNumber(int mapOutNumber){
        this.mapOutNumber = mapOutNumber;
    }

    public void setMapOutValue(int indexMapOut, int index, int i, int j, double value){
        if (indexMapOut >= mapOut.size()){
            System.out.printf("Ошибка. Индекс изображения превышает размер списка изображений (%d >= %d)", indexMapOut, mapOut.size());
            System.exit(-1);
        }

        if (index >= mapOut.get(indexMapOut).size()){
            System.out.printf("Ошибка. Индекс карты превышает размер списка карт (%d >= %d)", index, mapOut.get(indexMapOut).size());
            System.exit(-1);
        }

        if (i >= mapOut.get(indexMapOut).get(index).getRowNum()){
            System.out.printf("Ошибка. Индекс строки карты больше размера строк карты (%d >= %d)", i, mapOut.get(indexMapOut).get(index).getRowNum());
            System.exit(-1);
        }

        if (j >= mapOut.get(indexMapOut).get(index).getColNum()){
            System.out.printf("Ошибка. Индекс столбца карты больше размера столбца карты (%d >= %d)", j, mapOut.get(indexMapOut).get(index).getColNum());
            System.exit(-1);
        }

        List<Matrix> matrixes = mapOut.get(indexMapOut);
        Matrix matrix = matrixes.get(index);
        matrix.setValue(i,j, value);
    }

    public void setMapOutValue(int indexMapOut, int index, Matrix matrix){
        mapOut.get(indexMapOut).set(index, matrix);
    }

    public LayerType getType(){
        return type;
    }

    public int getMapOutNumber(){
        return mapOutNumber;
    }

    public Matrix getMap(int indexMapOut, int index){
        return mapOut.get(indexMapOut).get(index);
    }

    public Matrix getKernel(int indexMapOut, int index){
        if (indexMapOut >= kernel.size()){
            System.out.printf("Ошибка. Индекс изображения превышает размер списка изображений (%d >= %d)", indexMapOut, kernel.size());
            System.exit(-1);
        }

        if (index >= kernel.get(indexMapOut).size()){
            System.out.printf("Ошибка. Индекс ядра превышает размер списка карт (%d >= %d)", index, kernel.get(indexMapOut).size());
            System.exit(-1);
        }
        return kernel.get(indexMapOut).get(index);
    }

    public double getT(int index){
        return t[index];
    }

    public Matrix getError(int indexMapOut, int index){
        return error.get(indexMapOut).get(index);
    }

    public List<List<Matrix>> getError(){
        return error;
    }

    public Size getMapsSize(){
        return mapsSize;
    }

    public Size getKernelSize(){
        return kernelSize;
    }

    public Size getCompressSise(){
        return compressSise;
    }

    public int getClassNum(){
        return classNum;
    }
}