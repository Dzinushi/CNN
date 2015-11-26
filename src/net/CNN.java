package net;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import dataset.Mnist;
import util.*;


public class CNN {

    private int batchsize;
    private List<Layer> layers;
    private Precision precision;

    public CNN(){
        batchsize = 0;
        layers = new ArrayList<>();
        precision = new Precision();
    }

    // Инициирование параметров сети
    public void setup(CreateLayer layers, int batchSize){
        this.batchsize = batchSize;
        this.layers = layers.getListLayers();

        for (int i = 0; i < this.layers.size(); i++) {

            Layer inputLayer = null, layer;
            if (!Objects.equals(i,0)) {
                inputLayer = this.layers.get(i - 1);
            }
            layer = this.layers.get(i);

            switch (layer.getType()){

                case INPUT:
                    layer.setMapOut(batchSize);
                    break;

                case CONVOLUTION:
                    layer.setMapSize(inputLayer.getMapsSize().subtract(layer.getKernelSize(), 1));
                    layer.setKernel(inputLayer.getMapOutNumber());
                    layer.setT(inputLayer.getMapOutNumber());
                    layer.setError(this.batchsize);
                    layer.setMapOut(this.batchsize);
                    break;

                case SUBSAMPLING:
                    layer.setMapOutNumber(inputLayer.getMapOutNumber());
                    layer.setMapSize(inputLayer.getMapsSize().divide(layer.getCompressSise()));
                    layer.setError(this.batchsize);
                    layer.setMapOut(this.batchsize);
                    break;

                case OUTPUT:
                    layer.setOutKernel(inputLayer.getMapOutNumber(), inputLayer.getMapsSize());
                    layer.setT(inputLayer.getMapOutNumber());
                    layer.setError(this.batchsize);
                    layer.setMapOut(this.batchsize);
                    break;
            }
        }
    }

    /* Обучение сети. Вычисляется количество итераций, равное (количеству изображений БД) / batchsize.
    *  Формируем массив случайных индексов размер которого равна количеству изображений в БД.
    *  Производим последовательное обучение каждого слоя, подавая изображения в случайном порядке.
    * */
    public void train(Mnist mnist, int iteration){
        int numbatches = mnist.getSize() / batchsize;

        for (int i = 0; i < iteration; i++) {
            int[] randIndexes = Util.randPerm(mnist.getSize());

            for (int j = 0; j < numbatches; j++) {
                int index = randIndexes[i * batchsize + j];
                double[] image = mnist.getData(index);
                double[] lable = mnist.getLable(index);
                Size imageSize = new Size(mnist.getImageWidth(), mnist.getImageHeight());

                trainAllLayers(image, lable, imageSize, j);

                boolean right = backPropogation(image, lable, j);
                if (right)  precision.increase();
                LogCNN.printPrecision(getPrecision());
            }
        }
    }

    // Метод распостранения ошибки сети
    // Дописать рассчет ошибок выходного и скрытого слоев
    private boolean backPropogation(double[] data, double[] lable, int indexMapOut){
        Layer outLayer = layers.get(layers.size() - 1); // выходной слой
        double[] mapsOut = new double[outLayer.getMapOutNumber()];

        for (int i = 0; i < mapsOut.length; i++) {
            MapCNN mapOut = outLayer.getMap(indexMapOut, i);
            mapsOut[i] = mapOut.getValue(0,0);
        }

        return false;
    }

    // Обучение всех слоев нейронной сети
    private void trainAllLayers(double[] data, double[] lable, Size imageSize, int indexMapOut){
        for (int i = 0; i < layers.size(); i++) {

            Layer layer = null, layerPrev = null;
            if (!Objects.equals(i, 0)){
                layer = layers.get(i);
                layerPrev = layers.get(i - 1);
            }

            switch (layers.get(i).getType()){
                case INPUT:
                    trainInputLayer(data, lable, imageSize, indexMapOut);
                    break;

                case CONVOLUTION:
                    trainConvLayer(layer, layerPrev, indexMapOut);
                    break;

                case SUBSAMPLING:
                    trainSubLayer(layer, layerPrev, indexMapOut);
                    break;

                case OUTPUT:
                    trainOutLayer(layer, layerPrev, indexMapOut);
                    break;
            }
        }
    }

    // Задание изображения на входной слой (всего может быть batchsize изображений на входном слое)
    private void trainInputLayer(double[] data, double[] lable, Size imageSize, int indexMapOut){
        Layer layer = layers.get(0);
        for (int i = 0; i < imageSize.x; i++) {
            for (int j = 0; j < imageSize.y; j++) {
                layer.setMapOutValue(indexMapOut, 0, i, j, data[layer.getMapsSize().x * i + j]);
            }
        }
    }

    // Зачем использовать два layer?
    // Обучение сверточного слоя
    private void trainConvLayer(Layer layer, Layer layerPrev, int indexMapOut){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            MapCNN s = null;
            for (int j = 0; j < layerPrev.getMapOutNumber(); j++) {
                s = sum(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i), s);
            }
            layer.setMapOutValue(indexMapOut, i, s);
        }
    }

    // Обучение субдескритизирующего слоя
    private void trainSubLayer(Layer layer, Layer layerPrev, int indexMapOut){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            MapCNN sampMatrix = Util.compression(layerPrev.getMap(indexMapOut, i), layer.getCompressSise());
            layer.setMapOutValue(indexMapOut, i, sampMatrix);
        }
    }

    // Обучение выходного слоя
    private void trainOutLayer(Layer layer, Layer layerPrev, int indexMapOut){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            MapCNN s = null;
            for (int j = 1; j < layerPrev.getMapOutNumber(); j++) {
                s = sum(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i), s);
            }
            layer.setMapOutValue(indexMapOut, i, s);
        }
    }

    // Считаем размер карты исходя из размера ядра обхода изображения
    // Считаем взвешенную сумму для одной карты (функция активации гиперболический тангенс)
    private MapCNN sum(MapCNN image, MapCNN kernel, MapCNN currentSum){
        int row = image.getRowNum() - kernel.getRowNum() + 1;
        int column = image.getColNum() - kernel.getColNum() + 1;
        MapCNN result = new MapCNN(new Size(row, column));

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                double value = 0;
                for (int k = 0; k < kernel.getRowNum(); k++) {
                    for (int l = 0; l < kernel.getColNum(); l++) {
                        value += image.getValue(i+k, j+l) * kernel.getValue(k,l);
                    }
                }

                if (currentSum != null){
                    value += currentSum.getValue(i,j);
                    value = ActivationFunction.hyptan(value);
                    result.setValue(i, j, value);
                }
                else{
                    result.setValue(i, j, value);
                }
            }
        }

        return result;
    }

    private Precision getPrecision(){
        return precision;
    }
}