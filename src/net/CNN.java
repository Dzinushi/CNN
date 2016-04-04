package net;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import dataset.DataBase;
import util.*;

public class CNN implements Serializable{

    private int batchsize;
    private List<Layer> layers;
    private double lambda;
    private double alpha;
    private TimeCNN timeTraining;
    private Precision precision;

    private String name;
    private boolean autosave;

    private boolean usingAutoencoder;
    private int autoencoderIteration;

    private static final long serialVersionUID = -2606860604729216288L;

    // Задано константное значение alpha
    public CNN(){
        batchsize = 0;
        lambda = 0;
        alpha = 0.8;
        layers = new ArrayList<>();
        timeTraining = new TimeCNN();
        name = "";
        autosave = false;
        usingAutoencoder = false;
        autoencoderIteration = 0;
    }

    /**
     * Инициализация всех слоев сети
     * @param layers - список всех слоев сети
     * @param batchSize - размер выборки группового обучения
     */
    public void setup(CreateLayer layers, int batchSize){
        this.batchsize = batchSize;
        this.layers = layers.getListLayers();
        Layer layerPrev = null;

        for (Layer layer : this.layers) {
            switch (layer.getType()) {

                case INPUT:
                    layer.setMapOutSize(batchSize);
                    break;

                case CONVOLUTION:
                    layer.setMapSize(layerPrev.getMapsSize().subtract(layer.getKernelSize(), 1));
                    layer.setKernelSize(layerPrev.getMapOutNumber());
                    layer.setTSize();
                    layer.setErrorSize(this.batchsize);
                    layer.setMapOutSize(this.batchsize);
                    break;

                case SUBSAMPLING:
                    layer.setMapOutNumber(layerPrev.getMapOutNumber());
                    layer.setMapSize(layerPrev.getMapsSize().divide(layer.getCompressSise()));
                    layer.setErrorSize(this.batchsize);
                    layer.setMapOutSize(this.batchsize);
                    break;

                case OUTPUT:
                    layer.setOutKernel(layerPrev.getMapOutNumber(), layerPrev.getMapsSize());
                    layer.setTSize();
                    layer.setErrorSize(this.batchsize);
                    layer.setMapOutSize(this.batchsize);
                    break;
            }
            layerPrev = layer;
        }
    }

    /**
     * Обучение сети. Вычисляется количество итераций, равное (количеству изображений БД) / batchsize.
     *  Формируем массив случайных индексов размер которого равна количеству изображений в БД.
     *  Производим последовательное обучение каждого слоя, подавая изображения в случайном порядке.
     * @param trainData - база данных для обучения сети
     * @param iteration - количество итераций обучения
     */
    public void train(DataBase trainData, DataBase testData, int iteration){

        int numbatches = trainData.getSize() / batchsize;

        // если активирован автоэнкодер, то запустить его на всех изображениях, подаваемых для обучения сети
        if (usingAutoencoder) {
            trainWithAutoencoder(50, numbatches, trainData);
        }
        // обучение нейронной сети
        trainNet(iteration, numbatches, trainData, testData);
        LogCNN.printAllTime(this.timeTraining.getTimeAll());
    }

    private void trainWithAutoencoder(int iteration, int numbatches, DataBase trainData){
        System.out.println("Start study net using autoencoder");


        double[][] imageGroupList = trainData.getData();
        Size imageSize = new Size(trainData.getImageWidth(), trainData.getImageHeight());

       Autoencoder autoencoder = new Autoencoder();
        for (int i = autoencoderIteration; i < iteration; i++) {

            this.timeTraining.start();
            trainAllLayers(imageGroupList, imageSize, autoencoder);

            System.out.printf("%d from %d\tTime: %.3fs\n",
                    i+1,
                    iteration,
                    this.timeTraining.getTimeLast() / 1000.0);

            // сохраняем состояние сети после каждой итерации
            try {
                autoencoderIteration++;
                save(name);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        setUsingAutoencoder(false);
    }

    private void trainNet(int iteration, int numbatches, DataBase trainData, DataBase testData){
        System.out.println("Start study neuron network");

        Precision precision = new Precision();
        precision.setCount(trainData.getSize());

        // Лучшая точность сети
        Precision bestPrecision = new Precision();

        // Запуск функции отлавливающей нажатие 'Enter' в консоли и прерывающей обучение
        StopTrain stop = new StopTrain();
        Thread thread = new Thread(stop);
        thread.start();

        int index;
        for (int i = 0; i < iteration & !stop.isEnd(); i++) {
            this.timeTraining.start();

            index = 0;
            for (int j = 0; j < numbatches; j++) {
                for (int k = 0; k < batchsize; k++) {
                    double[] image = trainData.getData(index);
                    double[] label = trainData.getLabel(index);
                    Size imageSize = new Size(trainData.getImageHeight(), trainData.getImageWidth());

                    trainAllLayers(image, imageSize, k);
                    boolean right = backPropagation(label, k);

                    if (right){
                        precision.increase();
                    }
                    index++;
                    if (index == 8625){
                        index = 8625;
                    }
                }
                update();
            }

            // Проверка на тестируемой выборке. Сохранение лучшего результата
            Precision testPrecision = test(testData);
            if (testPrecision.getValue() > bestPrecision.getValue() && autosave){
                bestPrecision = testPrecision;
                try {
                    save(name);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            LogCNN.printTrainInfo(precision, testPrecision, timeTraining.getTimeLast());
            precision.resetValue();
        }
    }

    // обучение всех групповых элементов с ипользованием автоэнкодера
    private void trainAllLayers(double[][] data, Size imageSize, Autoencoder autoencoder){
        Layer layerPrev = null;
        int j = 0;

        while ( j < data.length ) {
            int nRunAutoencoder = 0;
            for (Layer layer : layers) {
                switch (layer.getType()){
                    case INPUT:
                        for (int i = 0; i < this.batchsize; i++) {
                            trainInputLayer(data[j], imageSize, i);
                            ++j;
                        }
                        break;

                    case CONVOLUTION:
                        for (int i = 0; i < this.batchsize; i++) {
                            trainConvLayer(layer, layerPrev, i);
                        }
                        autoencoder.start(layer, layerPrev, this.batchsize, nRunAutoencoder);
                        nRunAutoencoder++;
                        break;

                    case SUBSAMPLING:
                        for (int i = 0; i < this.batchsize; i++) {
                            trainSubLayer(layer, layerPrev, i);
                        }
                        break;

                    case OUTPUT:
                        for (int i = 0; i < this.batchsize; i++) {
                            trainOutLayer(layer, layerPrev, i);
                        }
                }
                layerPrev = layer;
            }
        }
    }

    /**
     * Обучение всех слоев сети
     * @param data - изображение из базы данных
     * @param imageSize - размер изображения
     * @param indexMapOut - индекс элемента группового обучения
     */
    private void trainAllLayers(double[] data, Size imageSize, int indexMapOut){
        Layer layerPrev = null;
        for (Layer layer : layers) {

            switch (layer.getType()){
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

            layerPrev = layer;
        }
    }

    /**
     * Формирование входного слоя на основе передаваемого изображения из базы данных
     * @param data - изображение из базы данных
     * @param imageSize - размер изображения
     * @param indexMapOut - индекс элемента группового обучения
     */
    private void trainInputLayer(double[] data, Size imageSize, int indexMapOut){
        Layer layer = layers.get(0);
        for (int i = 0; i < imageSize.x; i++) {
            for (int j = 0; j < imageSize.y; j++) {
                layer.setMapOutValue(indexMapOut, 0, i, j, data[imageSize.y * i + j]);
            }
        }
    }

    /**
     * Обучение сверточного слоя
     * @param layer - текущий сверточный слой
     * @param layerPrev - предыдущий слой (входной / субдескритизирующий)
     * @param indexMapOut - индекс элемента группового обучения
     */
    private void trainConvLayer(final Layer layer, final Layer layerPrev, int indexMapOut){
        TaskToThread taskToThread = new TaskToThread(layer.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int i = start; i < end; i++) {
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
                    s = activation(s, layer.getT(i), ActivationFunction.function.SIGM);
                    layer.setMapOutValue(indexMapOut, i, s);
                }
            }
        };
    }

    /**
     * Обучение субдескритизирующего слоя
     * @param layer - текущий субдескритизирующий слой
     * @param layerPrev - предыдущий слой (сверточный слой)
     * @param indexMapOut - индекс элемента группового обучения
     */
    private void trainSubLayer(final Layer layer, final Layer layerPrev, int indexMapOut){
        TaskToThread taskToThread = new TaskToThread(layer.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int i = start; i < end; i++) {
                    Matrix sampMatrix = MatrixOperation.compression(layerPrev.getMap(indexMapOut, i), layer.getCompressSise());
                    layer.setMapOutValue(indexMapOut, i, sampMatrix);
                }
            }
        };
    }

    /**
     * Обучение выходного слоя
     * @param layer - выходной слой
     * @param layerPrev - предыдущий слой (субдескритизирующий слой)
     * @param indexMapOut - индекс элемента группового обучения
     */
    private void trainOutLayer(Layer layer, Layer layerPrev, int indexMapOut){
        trainConvLayer(layer, layerPrev, indexMapOut);
    }

    /**
     * Вычисление взвешенной суммы как произведение значения карты предыдущего слоя с ядром свертки текущего
     * @param map - матрица карты предыдущего слоя
     * @param kernel - матрица ядра свертки текущего слоя
     * @return - матрица взвешенной суммы
     */
    private Matrix sum(final Matrix map, final Matrix kernel){
        int row = map.getRowNum() - kernel.getRowNum() + 1;
        int column = map.getColNum() - kernel.getColNum() + 1;
        Matrix result = new Matrix(new Size(row, column));

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                double value = 0.0;
                for (int k = 0; k < kernel.getRowNum(); k++) {
                    for (int l = 0; l < kernel.getColNum(); l++) {
                        value += map.getValue(i+k, j+l) * kernel.getValue(k,l);
                    }
                }

                result.setValue(i, j, value);
            }
        }

        return result;
    }

    /**
     * Рассчет функции активации для взвешенной суммы
     * @param s - матрица взвешенной суммы
     * @param tValue - пороговое значение
     * @return - результат апроксимации на кривую (сигмоида или гиперболический тангенс)
     */
    private Matrix activation(final Matrix s, double tValue, ActivationFunction.function functionName){
        for (int i = 0; i < s.getRowNum(); i++) {
            for (int j = 0; j < s.getColNum(); j++) {
				double value = s.getValue(i,j) + tValue;
                s.setValue(i, j, ActivationFunction.activation(functionName, value));
            }
        }
        return s;
    }

    /**
     * Вычисление ошибки обратного распространения
     * @param label - эталонное значение выхода
     * @param indexMapOut - индекс элемента группового обучения
     * @return - результат сравнения эталонного значения со значением полученным от сети
     */
    private boolean backPropagation(double[] label, int indexMapOut){

        boolean isRight = false;

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
                    isRight = calcOutErrors(label, indexMapOut);
                    break;
            }
        }

        return isRight;
    }

    /**
     * Вычисление ошибки сверточного слоя
     * @param layer - текущий субдескритизирующий слой
     * @param layerNext - следующий за ним слой - серточный / выходной
     * @param indexMapOut - индекс элемента группового обучения
     */
    private void calcConvErrors(final Layer layer, final Layer layerNext, int indexMapOut){
        TaskToThread taskToThread = new TaskToThread(layer.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int i = start; i < end; i++) {
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
        };
    }


    /**
     * Вычисление ошибки субдескритизирующего слоя
     * @param layer - текущий субдескритизирующий слой
     * @param layerNext - следующий за ним слой - серточный / выходной
     * @param indexMapOut - индекс элемента группового обучения
     */
    private void calcSubErrors(final Layer layer, final Layer layerNext, int indexMapOut){
        TaskToThread taskToThread = new TaskToThread(layer.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int i = start; i < end; i++) {
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

                    // нормализация ошибки с использованием RELU
//                    for (int j = 0; j < s.getRowNum(); j++) {
//                        for (int k = 0; k < s.getColNum(); k++) {
//                            if (s.getValue(j,k) <= 0){
//                                s.setValue(j,k,0);
//                            }
//                            else {
//                                s.setValue(j,k,1);
//                            }
//                        }
//                    }

                    layer.setErrorMap(indexMapOut, i, s);
                }
            }
        };
    }

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

    /**
     *
     * @param label - эталонное значение выхода
     * @param indexMapOut - номер элемента группового обучения
     * @return сравнение полученного от сети значения с эталонным
     */
    private boolean calcOutErrors(double[] label, int indexMapOut){
        Layer outLayer = layers.get(layers.size() - 1); // выходной слой
        double[] answer = new double[outLayer.getMapOutNumber()];

        for (int i = 0; i < answer.length; i++) {
            Matrix mapOut = outLayer.getMap(indexMapOut, i);
            answer[i] = mapOut.getValue(0,0);
        }

        int mapSize = outLayer.getMapOutNumber();
        for (int i = 0; i < mapSize; i++) {
            double value = answer[i] * (1 - answer[i]) * (label[i] - answer[i]);
            outLayer.setErrorValue(indexMapOut, i, 0, 0, value);
        }

        return isRightLabel(answer, label);
    }

    /**
     * Сравнение полученного от нейронной сети значения с эталонным значением и возврат результата в виде true / false
     * @param answer - результат распознавания полученный от нейронной сети
     * @param label - эталонное значение выхода
     * @return - сравнение полученного от сети значения с эталонным
     */
    private boolean isRightLabel(double[] answer, double[] label){
        int labelIndex = 0;
        boolean repeat = true;

        for (int i = 0; i < label.length & repeat; i++) {
            if (Objects.equals(label[i], 1.0)){
                labelIndex = i;
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
        return Objects.equals(labelIndex, answerIndex);
    }

    /**
     * Обновление ядер свертки и пороговых значений для сверточных и выходного слоев
     */
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

    /**
     * Обновление ядра свертки для сверточного / выходного слоев
     * @param layer - текущий сверточный / выходной слой
     * @param layerPrev - предыдущий слой (входной / субдескритизирующий)
     */
    private void updateKernel(final Layer layer, final Layer layerPrev){
        final int batchsize = this.batchsize;
        final double lambda = this.lambda;
        final double alpha = this.alpha;

        TaskToThread threadTaskManager = new TaskToThread(layer.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < layerPrev.getMapOutNumber(); j++) {
                        Matrix kernelNew = null;

                        for (int k = 0; k < batchsize; k++) {
                            if (kernelNew != null){
                                Matrix kernelNewCur = sum(layerPrev.getMap(k, j), layer.getError(k, i));
                                kernelNew = MatrixOperation.operation(kernelNew, kernelNewCur, MatrixOperation.Op.SUM);
                            }
                            else {
                                kernelNew = sum(layerPrev.getMap(k, j), layer.getError(k, i));
                            }
                        }

                        Matrix kernel = layer.getKernel(j,i);
                        kernel = MatrixOperation.multiply(kernel, 1.0 - lambda * alpha);
                        kernelNew = MatrixOperation.divide(kernelNew, batchsize);
                        kernelNew = MatrixOperation.multiply(kernelNew, alpha);

                        kernelNew = MatrixOperation.operation(kernel, kernelNew, MatrixOperation.Op.SUM);
                        layer.setKernelMatrix(j, i, kernelNew);
                    }
                }
            }
        };
    }

    /**
     * Обновление пороговых значений для сверточного и выходного слоев
     * @param layer - сверточный / выходной слой
     */
    private void updateT(final Layer layer){
        final int batchsize = this.batchsize;
        final double alpha = this.alpha;
        List<List<Matrix>> error  = layer.getError();

        TaskToThread taskToThread = new TaskToThread(layer.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int i = start; i < end; i++) {
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

                    double newValueT = MatrixOperation.sumAllCells(result) / batchsize;
                    newValueT = layer.getT(i) + alpha * newValueT;
                    layer.setTValue(i, newValueT);
                }
            }
        };
    }

    /**
     * Обучаем все слои нейронной сети на тестируемых данных, не изменяя весовые коэффициенты
     * и пороговые значения
     * @param testData - база данные для тестирования обученной нейронной сети
     * @return - возвращает объект класса содержащий количество распознанных данных и общее количество данных
     * для распознавания
     */
    public Precision test(DataBase testData){
        precision = new Precision();
        precision.setCount(testData.getSize());
        Size imageSize = new Size(testData.getImageHeight(), testData.getImageWidth());

        for (int i = 0; i < testData.getSize(); i++) {

            trainAllLayers(testData.getData(i), imageSize, 0);

            Layer layerOut = layers.get(layers.size() - 1);
            double[] answer = new double[layerOut.getMapOutNumber()];

            for (int j = 0; j < layerOut.getMapOutNumber(); j++) {
                answer[j] = layerOut.getMap(0, j).getValue(0,0);
            }

            boolean right = isRightLabel(answer, testData.getLabel(i));
            if (right){
                precision.increase();
            }

        }

        return precision;
    }


    /**
     * Сериализация объекта CNN в файл с именем "filename" и расширением ".cnn"
     * @param filename - имя файла сериализованного объекта CNN
     * @throws IOException - ошибка сохранения или открытия файла
     */
    public void save(String filename) throws IOException {
        FileOutputStream fos = new FileOutputStream(filename + ".cnn");
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(this);
        oos.flush();
        oos.close();
    }

    /**
     * Десериализация CNN-объекта из файла лежащего по пути "filepath" в объект среды.
     * @param filepath - путь к сеарилизованному объекту CNN
     * @return - десериализованный объект CNN
     * @throws IOException - ошибка открытия / чтения файла
     * @throws ClassNotFoundException - несуществующий класс
     */
    public CNN read(String filepath) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(filepath + ".cnn");
        ObjectInputStream oin = new ObjectInputStream(fis);
        return (CNN) oin.readObject();
    }

    public void netName(String string){
        this.name = string;
    }

    public Precision getPrecision(){
        return precision;
    }

    public int getBatchsize(){
        return batchsize;
    }

    public List<Layer> getLayers(){
        return layers;
    }

    public double getLambda(){
        return lambda;
    }

    public double getAlpha(){
        return alpha;
    }

    public void setName(String name){
        this.name = name;
    }

    public boolean isAutosave() {
        return autosave;
    }

    public void autosave(boolean autosave) {
        this.autosave = autosave;
    }

    public boolean isUsingAutoencoder() {
        return usingAutoencoder;
    }

    public void setUsingAutoencoder(boolean usingAutoencoder) {
        this.usingAutoencoder = usingAutoencoder;
    }

    public void setAlpha(double alpha){
        this.alpha = alpha;
    }
}