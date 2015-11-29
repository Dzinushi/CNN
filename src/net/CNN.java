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
    private double lambda;
    private double alpha;
    private StopTrain stop;

    // Задано константное значение alpha
    public CNN(){
        batchsize = 0;
        lambda = 0;
        alpha = 0.5;
        layers = new ArrayList<>();
        precision = new Precision();
        stop = new StopTrain();
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
                    layer.setMapOutSize(batchSize);
                    break;

                case CONVOLUTION:
                    layer.setMapSize(inputLayer.getMapsSize().subtract(layer.getKernelSize(), 1));
                    layer.setKernelSize(inputLayer.getMapOutNumber());
                    layer.setTSize();
                    layer.setErrorSize(this.batchsize);
                    layer.setMapOutSize(this.batchsize);
                    break;

                case SUBSAMPLING:
                    layer.setMapOutNumber(inputLayer.getMapOutNumber());
                    layer.setMapSize(inputLayer.getMapsSize().divide(layer.getCompressSise()));
                    layer.setErrorSize(this.batchsize);
                    layer.setMapOutSize(this.batchsize);
                    break;

                case OUTPUT:
                    layer.setOutKernel(inputLayer.getMapOutNumber(), inputLayer.getMapsSize());
                    layer.setTSize();
                    layer.setErrorSize(this.batchsize);
                    layer.setMapOutSize(this.batchsize);
                    break;
            }
        }
    }

    /* Обучение сети. Вычисляется количество итераций, равное (количеству изображений БД) / batchsize.
    *  Формируем массив случайных индексов размер которого равна количеству изображений в БД.
    *  Производим последовательное обучение каждого слоя, подавая изображения в случайном порядке.
    * */
    public void train(Mnist mnist, int iteration){
        TimeCNN timeCNN = new TimeCNN();
        timeCNN.start();
        int numbatches = mnist.getSize() / batchsize;
        this.precision.setCount(mnist.getSize());

        // Запус функции отлавливающей нажатие 'Enter' в консоли и прерывающей обучение
        Thread thread = new Thread(stop);
        thread.start();

        for (int i = 0; i < iteration & !this.stop.isEnd(); i++) {
            int[] randIndexes = Util.randPerm(mnist.getSize());

            for (int j = 0; j < numbatches; j++) {
                for (int k = 0; k < batchsize; k++) {
                    int index = randIndexes[j * batchsize + k];
                    double[] image = mnist.getData(index);
                    double[] lable = mnist.getLable(index);
                    Size imageSize = new Size(mnist.getImageWidth(), mnist.getImageHeight());

                    trainAllLayers(image, lable, imageSize, k);
                    boolean right = backPropagation(lable, k);

                    if (right){
                        precision.increase();
                    }
                }

                updateTandKernel();
            }

            LogCNN.printInfo(i+1, getPrecision(), timeCNN.getTimeLast());
            timeCNN.start();
            precision.resetValue();
        }

        LogCNN.printAllTime(timeCNN.getTimeAll());
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
                    trainInputLayer(data, imageSize, indexMapOut);
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
    private void trainInputLayer(double[] data, Size imageSize, int indexMapOut){
        Layer layer = layers.get(0);
        for (int i = 0; i < imageSize.x; i++) {
            for (int j = 0; j < imageSize.y; j++) {
                layer.setMapOutValue(indexMapOut, 0, i, j, data[layer.getMapsSize().x * i + j]);
            }
        }
    }

    // Обучение сверточного слоя
    private void trainConvLayer(final Layer layer, final Layer layerPrev, int indexMapOut){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            Matrix s = null;
            for (int j = 0; j < layerPrev.getMapOutNumber(); j++) {
                s = sum(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i), s);
            }
            s = activation(s, layer.getT(i));
            layer.setMapOutValue(indexMapOut, i, s);
        }
    }

    // Обучение субдескритизирующего слоя
    private void trainSubLayer(final Layer layer, final Layer layerPrev, int indexMapOut){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            Matrix sampMatrix = MatrixOperation.compression(layerPrev.getMap(indexMapOut, i), layer.getCompressSise());
            layer.setMapOutValue(indexMapOut, i, sampMatrix);
        }
    }

    // Обучение выходного слоя
    private void trainOutLayer(Layer layer, Layer layerPrev, int indexMapOut){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            Matrix s = null;
            for (int j = 0; j < layerPrev.getMapOutNumber(); j++) {
                s = sum(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i), s);
            }
            s = activation(s, layer.getT(i));
            layer.setMapOutValue(indexMapOut, i, s);
        }
    }

    // Считаем размер карты исходя из размера ядра обхода изображения
    // Считаем взвешенную сумму для одной карты (функция активации сигмоидная)
    private Matrix sum(final Matrix image, final Matrix kernel, final Matrix currentSum){
        int row = image.getRowNum() - kernel.getRowNum() + 1;
        int column = image.getColNum() - kernel.getColNum() + 1;
        Matrix result = new Matrix(new Size(row, column));

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
                    result.setValue(i, j, value);
                }
                else{
                    result.setValue(i, j, value);
                }
            }
        }

        return result;
    }

    // Рассчет функции активации для взвешенной суммы
    private Matrix activation(final Matrix s, double tValue){
        for (int i = 0; i < s.getRowNum(); i++) {
            for (int j = 0; j < s.getColNum(); j++) {
                s.setValue(i, j, ActivationFunction.sigm(s.getValue(i,j)) + tValue);
            }
        }
        return s;
    }

    // Метод распостранения ошибки сети
    // Дописать рассчет ошибок выходного и скрытого слоев
    private boolean backPropagation(double[] lable, int indexMapOut){

        boolean isRight = false;

        // Ошибки сверточных и субдескритизирующих слоев
        // 0-й слой - входной слой, не имеющий ошибок, содержащий эталонные значения
        for (int i = layers.size() - 1; i > 0; i--) {
            Layer layer = layers.get(i);
            Layer layerNext = null;
            if (!Objects.equals(i, layers.size() - 1)){
                layerNext = layers.get(i+1);
            }

            switch (layer.getType()){

                case CONVOLUTION:
                    calcConvErrors(layer, layerNext, indexMapOut);
                    break;

                case SUBSAMPLING:
                    calcSubErrors(layer, layerNext, indexMapOut);
                    break;

                case OUTPUT:
                    isRight = calcOutErrors(lable, indexMapOut);
                    break;
            }
        }

        return isRight;
    }

    private void calcConvErrors(final Layer layer, final Layer layerNext, int indexMapOut){
        Matrix mapError, map, s, increaseMap;

        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            mapError = layerNext.getError(indexMapOut, i);
            map = layer.getMap(indexMapOut, i);
            Matrix mapClone = MatrixOperation.clone(map);
            mapClone = MatrixOperation.valMinus(mapClone, 1.0);
            s = MatrixOperation.operation(map, mapClone, MatrixOperation.Op.MULTIPLY);

            increaseMap = MatrixOperation.extend(mapError, layerNext.getCompressSise());
            s = MatrixOperation.operation(s, increaseMap, MatrixOperation.Op.MULTIPLY);

            layer.setErrorMap(indexMapOut, i, s);
        }
    }

    // Вычисление ошибки субдескритизирующего слоя
    private void calcSubErrors(final Layer layer, final Layer layerNext, int indexMapOut){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            Matrix mapError, kernel, s = null;
            for (int j = 0; j < layerNext.getMapOutNumber(); j++) {
                mapError = layerNext.getError(indexMapOut, j);
                kernel = layerNext.getKernel(i, j);
                s = convFull(mapError, MatrixOperation.rot180(kernel), s);
            }
            layer.setErrorMap(indexMapOut, i, s);
        }
    }

    // Вычисление взвешенной суммы сверточного слоя
    private Matrix convFull(final Matrix matrix, final Matrix kernel, final Matrix s){
        int row = matrix.getRowNum() + 2 * (kernel.getRowNum() - 1);
        int column = matrix.getColNum() + 2 * (kernel.getColNum() - 1);
        Matrix extend = new Matrix(new Size(row, column));

        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                extend.setValue(i + kernel.getRowNum() - 1, j + kernel.getColNum() - 1, matrix.getValue(i,j));
            }
        }

        return sum(extend, kernel, s);
    }

    // Ошибки выходного слоя
    private boolean calcOutErrors(double[] lable, int indexMapOut){
        Layer outLayer = layers.get(layers.size() - 1); // выходной слой
        double[] mapsOut = new double[outLayer.getMapOutNumber()];

        for (int i = 0; i < mapsOut.length; i++) {
            Matrix mapOut = outLayer.getMap(indexMapOut, i);
            mapsOut[i] = mapOut.getValue(0,0);
        }

        int mapSize = outLayer.getMapOutNumber();
        for (int i = 0; i < mapSize; i++) {
            double value = mapsOut[i] * (1 - mapsOut[i]) * (lable[i] - mapsOut[i]);
            outLayer.setErrorValue(indexMapOut, i, 0, 0, value);
        }

        return isRightLable(mapsOut, lable);
    }

    private boolean isRightLable(double[] mapsOut, double[] lable){
        int lableIndex = 0;
        boolean repeat = true;

        for (int i = 0; i < lable.length & repeat; i++) {
            if (Objects.equals(lable[i], 1.0)){
                lableIndex = i;
                repeat = false;
            }
        }

        int lableIndexNet = 0;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < mapsOut.length; i++) {
            if (mapsOut[i] > max){
                lableIndexNet = i;
                max = mapsOut[i];
            }
        }
        return Objects.equals(lableIndex, lableIndexNet);
    }

    // Обновление значений ядра свертки и пороговых значений
    // 0-й слой - входной слой, не требующий обновления
    private void updateTandKernel(){
        for (int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            Layer layerPrev = layers.get(i-1);

            if (layer.getType() == Layer.LayerType.CONVOLUTION | layer.getType() == Layer.LayerType.OUTPUT){
                updateKernel(layer, layerPrev);
                updateT(layer);
            }
        }
    }

    private void updateKernel(final Layer layer, final Layer layerPrev){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            for (int j = 0; j < layerPrev.getMapOutNumber(); j++) {
                Matrix kernelNew = null;

                for (int k = 0; k < this.batchsize; k++) {
                    kernelNew = sum(layerPrev.getMap(k, j), layer.getError(k, i), kernelNew);
                }

                Matrix kernel = layer.getKernel(j,i);
                kernel = MatrixOperation.multiply(kernel, 1 - this.lambda * this.alpha);
                kernelNew = MatrixOperation.divide(kernelNew, this.batchsize);
                kernelNew = MatrixOperation.multiply(kernelNew, this.alpha);

                kernelNew = MatrixOperation.operation(kernel, kernelNew, MatrixOperation.Op.SUM);
                layer.setKernelMatrix(j, i, kernelNew);
            }
        }
    }

    private void updateT(final Layer layer){
        List<List<Matrix>> error  = layer.getError();
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            int row = error.get(i).get(0).getRowNum();
            int column = error.get(i).get(0).getColNum();
            Matrix result = new Matrix(new Size(row, column));

            for (int j = 0; j < row; j++) {
                for (int k = 0; k < column; k++) {
                    double value = 0;

                    for (List<Matrix> anError : error) {
                        value += anError.get(i).getValue(j, k);
                    }
                    result.setValue(j, k, value);
                }
            }

            double newValueT = MatrixOperation.sumAllCells(result) / this.batchsize;
            newValueT = layer.getT(i) + this.alpha * newValueT;
            layer.setTValue(i, newValueT);
        }
    }

    private Precision getPrecision(){
        return precision;
    }
}