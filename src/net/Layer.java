package net;

import util.MapCNN;
import util.Size;
import util.Util;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    public LayerType type;
    private int mapOutNumber;
    private Size mapsSize;
    private Size kernelSize;
    private Size compressSise;
    private int classNum = -1;

    private List<List<MapCNN>> kernel;  // ядра свертки
    private List<List<MapCNN>> error;   // ошибки карт
    private List<List<MapCNN>> mapOut;  // набор карт
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
    public void setKernel(int outMapNumber){
        kernel = new ArrayList<>();
        for (int i = 0; i < outMapNumber; i++) {
            List<MapCNN> mapCNNs = new ArrayList<>();
            for (int j = 0; j < getMapOutNumber(); j++) {
                MapCNN mapCNN = new MapCNN(getMapsSize());
                mapCNNs.add(mapCNN);
            }
            kernel.add(mapCNNs);
        }
    }

    // Пока без сдвига (сдвиг = 0)
    public void setT(int outMapNumber){
        t = new double[outMapNumber];
    }

    // Задать значение порога по индексу
    public void setTValue(int outMapNumber, double value){
        t[outMapNumber] = value;
    }

    // Заданем размерность ошибок
    public void setError(int batchsize){
        error = new ArrayList<>(batchsize);
        for (int i = 0; i < batchsize; i++) {
            List<MapCNN> mapCNNs = new ArrayList<>();
            for (int j = 0; j < getMapOutNumber(); j++) {
                MapCNN mapCNN = new MapCNN(getMapsSize());
                mapCNNs.add(mapCNN);
            }
            error.add(mapCNNs);
        }
    }

    // Задаем размерность карт на выходе
    public void setMapOut(int batchsize){
        mapOut = new ArrayList<>(batchsize);
        for (int i = 0; i < batchsize; i++) {
            List<MapCNN> mapCNNs = new ArrayList<>();
            for (int j = 0; j < getMapOutNumber(); j++) {
                MapCNN mapCNN = new MapCNN(getMapsSize());
                mapCNNs.add(mapCNN);
            }
            mapOut.add(mapCNNs);
        }
    }

    public void setMapOutNumber(int mapOutNumber){
        this.mapOutNumber = mapOutNumber;
    }

    // Задаем размерность ядра свертки для выходного слоя
    public void setOutKernel(int outMapNumber, Size mapSize){
        kernelSize = mapSize;
        setKernel(outMapNumber);

        for (int i = 0; i < outMapNumber; i++) {
            List<MapCNN> mapCNNs = kernel.get(i);
            for (int j = 0; j < getMapOutNumber(); j++) {
                mapCNNs.set(j, Util.randomMapCNN(mapSize));
            }
        }
    }

    public void setMapOutValue(int indexMapOut, int index, int i, int j, double value){
        List<MapCNN> mapCNNs = mapOut.get(indexMapOut);
        MapCNN mapCNN = mapCNNs.get(index);
        mapCNN.setValue(i,j, value);
    }

    public void setMapOutValue(int indexMapOut, int index, MapCNN mapCNN){
        mapOut.get(indexMapOut).set(index, mapCNN);
    }

    public LayerType getType(){
        return type;
    }

    public int getMapOutNumber(){
        return mapOutNumber;
    }

    public MapCNN getMap(int indexMapOut, int index){
        return mapOut.get(indexMapOut).get(index);
    }

    public MapCNN getKernel(int indexMapOut, int index){
        return kernel.get(indexMapOut).get(index);
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