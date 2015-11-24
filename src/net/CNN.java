package net;

import java.util.List;
import java.util.Objects;

import dataset.Mnist;
import util.Size;
import util.Util;


public class CNN {

    private int batchsize;
    private List<Layer> layers;

    // Инициирование параметров сети
    public void setup(CreateLayer layers, int batchSize){
        this.batchsize = batchSize;
        this.layers = layers.getListLayers();

        // Инициализация остальных слоёв
        for (int i = 0; i < this.layers.size(); i++) {

            Layer firstLayer = null, layer;
            if (!Objects.equals(i,0)) {
                firstLayer = this.layers.get(i - 1);
            }
            layer = this.layers.get(i);

            switch (layer.getType()){

                case INPUT:
                    layer.setMapOut(batchSize);
                    break;

                case CONVOLUTION:
                    layer.setMapSize(firstLayer.getMapsSize().subtract(layer.getKernelSize(), 1));
                    layer.setKernel(firstLayer.getMapOutNumber());
                    layer.setShift(firstLayer.getMapOutNumber());
                    layer.setError(this.batchsize);
                    layer.setMapOut(this.batchsize);
                    break;

                case SUBSAMPLING:
                    layer.setMapOutNumber(firstLayer.getMapOutNumber());
                    layer.setMapSize(firstLayer.getMapsSize().divide(layer.getScaleSize()));
                    layer.setError(this.batchsize);
                    layer.setMapOut(this.batchsize);
                    break;

                case OUTPUT:
                    layer.setOutKernel(firstLayer.getMapOutNumber(), firstLayer.getMapsSize());
                    layer.setShift(firstLayer.getMapOutNumber());
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

                trainAllLAyers(image, lable, imageSize, j);
                setOutLayerError(image, lable, j);
                backPropogation(image, lable);
            }
        }
    }

    private void backPropogation(double[] data, double[] lable){

    }

    private boolean setOutLayerError(double[] data, double[] lable, int indexMapOut){
        Layer outLayer = layers.get(layers.size() - 1); // выходной слой
        double[] mapsOut = new double[outLayer.getMapOutNumber()];

        for (int i = 0; i < mapsOut.length; i++) {
            MapCNN mapOut = outLayer.getMap(indexMapOut, i);
            mapsOut[i] = mapOut.getValue(0,0);
        }



        return false;
    }

    private void trainAllLAyers(double[] data, double[] lable, Size imageSize, int indexMapOut){
        for (int i = 0; i < layers.size(); i++) {

            Layer layer, lastLayer;
            if (!Objects.equals(i, 0)){
                layer = layers.get(i);
                lastLayer = layers.get(i - 1);
            }

            switch (layers.get(i).getType()){
                case INPUT:
                    setInputLayerOut(data, lable, imageSize, indexMapOut);
                    break;

                case CONVOLUTION:
                    break;

                case SUBSAMPLING:
                    break;

                case OUTPUT:
                    break;
            }
        }
    }

    private void setInputLayerOut(double[] data, double[] lable, Size imageSize, int indexMapOut){
        Layer layer = layers.get(0);
        for (int i = 0; i < imageSize.x; i++) {
            for (int j = 0; j < imageSize.y; j++) {
                double value = data[layer.getMapsSize().x * i + j];
                layer.setMapOutValue(indexMapOut, 0, i, j, value);
            }
        }
    }

    private void setConvLayerOut(){

    }

    private void setSubLayerOut(){

    }

    private void setOutputLayerOut(){

    }
}