package net;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
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

    private int count = 0;

    // Задано константное значение alpha
    public CNN(){
        batchsize = 0;
        lambda = 0;
        alpha = 0.85;
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

        System.out.printf("\nStart training\n");

        for (int i = 0; i < iteration & !this.stop.isEnd(); i++) {
            //int[] randIndexes = Util.randPerm(mnist.getSize());

            for (int j = 0; j < numbatches; j++) {

                int[] randIndexes = Util.randPerm(mnist.getSize(), batchsize);

                for (int k = 0; k < batchsize; k++) {
                    int index = randIndexes[k];
                    double[] image = mnist.getData(index);
                    double[] lable = mnist.getLable(index);
                    Size imageSize = new Size(mnist.getImageWidth(), mnist.getImageHeight());

                    trainAllLayers(image, imageSize, k);
                    boolean right = backPropagation(lable, k);

                    if (right){
                        precision.increase();
                    }

                    // удалить
                    //saveData();
                }

                update();
            }

            LogCNN.printInfo(getPrecision(), timeCNN.getTimeLast());
            timeCNN.start();
            precision.resetValue();
        }

        LogCNN.printAllTime(timeCNN.getTimeAll());
    }

    // Обучение всех слоев нейронной сети
    private void trainAllLayers(double[] data, Size imageSize, int indexMapOut){
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
                if (s != null){
                    Matrix sCur = sum(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i));
                    s = MatrixOperation.operation(s, sCur, MatrixOperation.Op.SUM);
                }
                else {
                    s = sum(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i));
                }
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
                if (s != null){
                    Matrix sCur = sum(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i));
                    s = MatrixOperation.operation(s, sCur, MatrixOperation.Op.SUM);
                }
                else {
                    s = sum(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i));
                }
            }
            s = activation(s, layer.getT(i));
            layer.setMapOutValue(indexMapOut, i, s);
        }
    }

    // Считаем размер карты исходя из размера ядра обхода изображения
    // Считаем взвешенную сумму для одной карты (функция активации сигмоидная)
    private Matrix sum(final Matrix image, final Matrix kernel){
        int row = image.getRowNum() - kernel.getRowNum() + 1;
        int column = image.getColNum() - kernel.getColNum() + 1;
        Matrix result = new Matrix(new Size(row, column));

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                double value = 0.0;
                for (int k = 0; k < kernel.getRowNum(); k++) {
                    for (int l = 0; l < kernel.getColNum(); l++) {
                        value += image.getValue(i+k, j+l) * kernel.getValue(k,l);
                    }
                }

                result.setValue(i, j, value);
            }
        }

        return result;
    }

    // Рассчет функции активации для взвешенной суммы
    private Matrix activation(final Matrix s, double tValue){
        for (int i = 0; i < s.getRowNum(); i++) {
            for (int j = 0; j < s.getColNum(); j++) {
                s.setValue(i, j, ActivationFunction.sigm(s.getValue(i,j) + tValue));
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
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            Matrix mapError = layerNext.getError(indexMapOut, i);
            Matrix map = layer.getMap(indexMapOut, i);
            Matrix mapClone = MatrixOperation.clone(map);
            mapClone = MatrixOperation.valMinus(mapClone, 1.0);
            Matrix s = MatrixOperation.operation(map, mapClone, MatrixOperation.Op.MULTIPLY);

            Matrix extendMap = MatrixOperation.extend(mapError, layerNext.getCompressSise());
            s = MatrixOperation.operation(s, extendMap, MatrixOperation.Op.MULTIPLY);

            layer.setErrorMap(indexMapOut, i, s);
        }
    }

    // Вычисление ошибки субдескритизирующего слоя
    private void calcSubErrors(final Layer layer, final Layer layerNext, int indexMapOut){
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            Matrix s = null;
            for (int j = 0; j < layerNext.getMapOutNumber(); j++) {
                Matrix mapError = layerNext.getError(indexMapOut, j);
                Matrix kernel = layerNext.getKernel(i, j);

                if (s != null){
                    Matrix sCur = calcMatrixConvError(mapError, MatrixOperation.rot180(kernel));
                    s = MatrixOperation.operation(s, sCur, MatrixOperation.Op.SUM);
                }
                else {
                    s = calcMatrixConvError(mapError, MatrixOperation.rot180(kernel));
                }
            }
            layer.setErrorMap(indexMapOut, i, s);
        }
    }

    // Вычисление взвешенной суммы сверточного слоя
    private Matrix calcMatrixConvError(final Matrix matrix, final Matrix kernel){
        int row = matrix.getRowNum() + 2 * (kernel.getRowNum() - 1);
        int column = matrix.getColNum() + 2 * (kernel.getColNum() - 1);
        Matrix extend = new Matrix(new Size(row, column));

        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                extend.setValue(i + kernel.getRowNum() - 1, j + kernel.getColNum() - 1, matrix.getValue(i,j));
            }
        }

        return sum(extend, kernel);
    }

    // Ошибки выходного слоя
    private boolean calcOutErrors(double[] lable, int indexMapOut){
        Layer outLayer = layers.get(layers.size() - 1); // выходной слой
        double[] answer = new double[outLayer.getMapOutNumber()];

        for (int i = 0; i < answer.length; i++) {
            Matrix mapOut = outLayer.getMap(indexMapOut, i);
            answer[i] = mapOut.getValue(0,0);
        }

        int mapSize = outLayer.getMapOutNumber();
        for (int i = 0; i < mapSize; i++) {
            double value = answer[i] * (1 - answer[i]) * (lable[i] - answer[i]);
            outLayer.setErrorValue(indexMapOut, i, 0, 0, value);
        }

        return isRightLable(answer, lable);
    }

    private boolean isRightLable(double[] answer, double[] lable){
        int lableIndex = 0;
        boolean repeat = true;

        for (int i = 0; i < lable.length & repeat; i++) {
            if (Objects.equals(lable[i], 1.0)){
                lableIndex = i;
                repeat = false;
            }
        }

        int answerIndex = 0;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < answer.length; i++) {
            if (answer[i] > max){
                answerIndex = i;
                max = answer[i];
            }
        }
        return Objects.equals(lableIndex, answerIndex);
    }

    // Обновление значений ядра свертки и пороговых значений
    // 0-й слой - входной слой, не требующий обновления
    private void update(){
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
                    if (kernelNew != null){
                        Matrix kernelNewCur = sum(layerPrev.getMap(k, j), layer.getError(k, i));
                        kernelNew = MatrixOperation.operation(kernelNew, kernelNewCur, MatrixOperation.Op.SUM);
                    }
                    else {
                        kernelNew = sum(layerPrev.getMap(k, j), layer.getError(k, i));
                    }
                }

                Matrix kernel = layer.getKernel(j,i);
                kernel = MatrixOperation.multiply(kernel, 1.0 - this.lambda * this.alpha);
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





    private void saveData(){
        PrintWriter printWriter = null;

        try {
            printWriter = new PrintWriter("D:\\Development\\Java\\PROJECTS\\University\\5\\Course\\CNN_for_Image\\temp\\" + String.valueOf(count) + "_CNN_for_image.txt", "UTF-8");
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        assert printWriter != null;

        for (Layer layer : this.layers) {
            switch (layer.getType()){
                case INPUT:
                    printWriter.println(layer.getType().toString());
                    printWriter.print(stringOutMaps(layer));
                    break;
                case CONVOLUTION:
                    printWriter.println(layer.getType().toString());
                    printWriter.print(stringOutMaps(layer));
                    printWriter.print(stringErrors(layer));
                    printWriter.print(stringKernels(layer));
                    printWriter.print(stringT(layer));
                    break;
                case SUBSAMPLING:
                    printWriter.println(layer.getType().toString());
                    printWriter.print(stringOutMaps(layer));
                    printWriter.print(stringErrors(layer));
                    break;
                case OUTPUT:
                    printWriter.println(layer.getType().toString());
                    printWriter.print(stringOutMaps(layer));
                    printWriter.print(stringErrors(layer));
                    printWriter.print(stringKernels(layer));
                    printWriter.print(stringT(layer));
                    break;
            }
        }

        count++;
        printWriter.close();
    }

    private String stringOutMaps(Layer layer){
        StringBuilder stringBuilder = new StringBuilder("\nMaps\n");
        for (int i = 0; i < batchsize; i++) {
            for (int j = 0; j < layer.getMapOutNumber(); j++) {
                Matrix matrix = layer.getMap(i,j);
                for (int k = 0; k < matrix.getRowNum(); k++) {
                    for (int l = 0; l < matrix.getColNum(); l++) {
                        stringBuilder.append('[')
                                .append(i)
                                .append(']')
                                .append('[')
                                .append(j)
                                .append(']')
                                .append('[')
                                .append(k)
                                .append(']')
                                .append('=')
                                .append(matrix.getValue(k,l))
                                .append('\n');
                    }
                }
            }
        }
        return stringBuilder.toString();
    }

    private String stringErrors(Layer layer){
        StringBuilder stringBuilder = new StringBuilder("\nErrors\n");
        for (int i = 0; i < batchsize; i++) {
            for (int j = 0; j < layer.getMapOutNumber(); j++) {
                Matrix matrix = layer.getError(i,j);
                for (int k = 0; k < matrix.getRowNum(); k++) {
                    for (int l = 0; l < matrix.getColNum(); l++) {
                        stringBuilder.append('[')
                                .append(i)
                                .append(']')
                                .append('[')
                                .append(j)
                                .append(']')
                                .append('[')
                                .append(k)
                                .append(']')
                                .append('=')
                                .append(matrix.getValue(k,l))
                                .append('\n');
                    }
                }
            }
        }
        return stringBuilder.toString();
    }

    private String stringKernels(Layer layer){
        StringBuilder stringBuilder = new StringBuilder("\nKernels\n");
        List<List<Matrix>> kernel = layer.getKernel();
        for (int i = 0; i < kernel.size(); i++) {
            for (int j = 0; j < kernel.get(i).size(); j++) {
                Matrix matrix = layer.getKernel(i,j);
                for (int k = 0; k < matrix.getRowNum(); k++) {
                    for (int l = 0; l < matrix.getColNum(); l++) {
                        stringBuilder.append('[')
                                .append(i)
                                .append(']')
                                .append('[')
                                .append(j)
                                .append(']')
                                .append('[')
                                .append(k)
                                .append(']')
                                .append('=')
                                .append(matrix.getValue(k,l))
                                .append('\n');
                    }
                }
            }
        }
        return stringBuilder.toString();
    }

    private String stringT(Layer layer){
        StringBuilder stringBuilder = new StringBuilder("\nT\n");
        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            stringBuilder.append('[')
                    .append(i)
                    .append(']')
                    .append('=')
                    .append(layer.getT(i))
                    .append('\n');
        }
        return stringBuilder.toString();
    }
}