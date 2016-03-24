package net;

import util.Matrix;
import util.MatrixOperation;
import util.Size;

import java.util.List;

class Autoencoder {

    private enum operation{
        ENCODE,
        DECODE
    }

    /**
     *
     * @param hidden        - сверточный слой
     * @param input         - входной / субдескритизирующий слой
     * @param batchsize     - количество элементов групповой выборки
     *
     * input -> hidden  -> out - первый прогон по автоэнкодеру
     * out   -> hidden1        - второй прогон по автоэнкодеру
     * input    = x(0)
     * hidden   = y(0)
     * out      = x(1)
     * hidden1  = y(1)
     */
    void start(Layer hidden, Layer input, int batchsize){

        Layer out = new Layer(input);
        Layer hidden1 = new Layer(hidden);

        for (int i = 0; i < batchsize; i++) {
            train(out, hidden, i, operation.DECODE);
            train(hidden1, out, i, operation.ENCODE);
        }

        // обновление ядра / ядер свертки на сверточном слое hidden
        update(input, hidden, out, hidden1, batchsize);
    }

    private void difference(Layer original, Layer recover, int indexMapOut){
        Matrix orMatrix = original.getMap(indexMapOut, 0);
        Matrix recMatrix = recover.getMap(indexMapOut, 0);
        double diff = 0.0;
        for (int i = 0; i < orMatrix.getRowNum(); i++) {
            for (int j = 0; j < orMatrix.getColNum(); j++) {
                diff += Math.abs(orMatrix.getValue(i,j) - recMatrix.getValue(i,j));
            }
        }
        System.out.println(diff);
    }

    private void train(Layer layer, Layer layerPrev, int indexMapOut, operation operation){

        for (int i = 0; i < layer.getMapOutNumber(); i++) {
            Matrix s = null;
            for (int j = 0; j < layerPrev.getMapOutNumber(); j++) {
                // если карта уже существует, то суммируем полученную и существующую карты
                if (s != null){
                    Matrix sCur;
                    if (layer.getType() == Layer.LayerType.INPUT || layer.getType() == Layer.LayerType.SUBSAMPLING){
                        sCur = operation(layerPrev.getMap(indexMapOut, j), layerPrev.getKernel(i, j), operation);
                    }
                    else {
                         sCur = operation(layerPrev.getMap(indexMapOut, j), layer.getKernel(j, i), operation);
                    }
                    s = MatrixOperation.operation(s, sCur, MatrixOperation.Op.SUM);
                }
                // если карты не существует, создаем карту
                else {
                    if (layer.getType() == Layer.LayerType.INPUT || layer.getType() == Layer.LayerType.SUBSAMPLING){
                        s = operation(layerPrev.getMap(indexMapOut, j), layerPrev.getKernel(i, j), operation);
                    }
                    else {
                        s = operation(layerPrev.getMap(indexMapOut, j), layer.getKernel(j, i), operation);
                    }
                }
            }
            // вычисляем значение карты с учетом порогов и функции активации
            if (layer.getType() == Layer.LayerType.INPUT || layer.getType() == Layer.LayerType.SUBSAMPLING){
                s = activation(s, layerPrev.getT(i), ActivationFunction.function.SIGM);
            }
            else {
                s = activation(s, layer.getT(i), ActivationFunction.function.SIGM);
            }
            layer.setMapOutValue(indexMapOut, i, s);
        }
    }

    private Matrix operation(final Matrix map, final Matrix kernel, operation operation){
        Matrix matrix;
        if (operation == Autoencoder.operation.ENCODE){
            matrix = encode(map, kernel);
        }
        else {
            matrix = decode(map, kernel);
        }
        return matrix;
    }

    private Matrix encode(final Matrix map, final Matrix kernel){
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

    private Matrix decode(final Matrix map, final Matrix kernel){
        int row = map.getRowNum() + kernel.getRowNum() - 1;
        int column = map.getColNum() + kernel.getColNum() - 1;
        Matrix result = new Matrix(new Size(row, column));

        for (int i = 0; i < map.getRowNum(); i++) {
            for (int j = 0; j < map.getColNum(); j++) {
                for (int k = 0; k < kernel.getRowNum(); k++) {
                    for (int l = 0; l < kernel.getColNum(); l++) {
                        double value = result.getValue(i+k, j+l);
                        value += map.getValue(i, j) * kernel.getValue(k,l);
                        result.setValue(i+k, j+l, value);
                    }
                }
            }
        }

        return result;
    }

    private Matrix activation(final Matrix s, double tValue, ActivationFunction.function functionName){
        for (int i = 0; i < s.getRowNum(); i++) {
            for (int j = 0; j < s.getColNum(); j++) {
                double value = s.getValue(i,j) + tValue;
                s.setValue(i, j, ActivationFunction.activation(functionName, value));
            }
        }
        return s;
    }

    // исправить строку "double value ..."
    private void update(Layer x0, Layer y0, Layer x1, Layer y1, int batchsize){

        // обновление весов
        // получение списка весовых коэффициентов для данного изображения
        // i - номер элемента групповой выбрки
        // j - номер карты элемента группповой выборки
        for (int i = 0; i < y0.getKernel().size(); i++) {

            List<Matrix> kernels = y0.getKernel().get(i);
            for (int j = 0; j < kernels.size(); ++j) {

                // формируем матрицу связей между входным и скрытым слоем, в которой число строк равно числу элементов входного слоя, а число столбцов числу элементов скрытого слоя
                Matrix differents = calcLinksLayers(x0, y0, x1, y1, batchsize, j);

                // без сложения одинаковых весовых коэффициентов
                Matrix kernel = kernels.get(j);
                int num = 0;
                for (int k = 0; k < kernel.getColNum(); k++) {
                    for (int l = 0; l < kernel.getRowNum(); l++) {
                        double sum = sumSimilarWeights(differents, num);
                        kernel.setValue(l, k, kernel.getValue(k,l) + ((0.5 / batchsize) * sum));
                        num++;
                    }
                }
            }
        }

        // обновление порогов
    }

    private double sumSimilarWeights(final Matrix matrix, int i){
        int j = 0;
        double sum = 0.0;
        int size = matrix.getColNum();

        while(j < size){
            sum += matrix.getValue(i,j);
            i++;
            j++;
        }
        return sum;
    }

    private double getValue(Matrix matrix, int curValue, Size kernelSize, int kernelX, int kernelY, int kernelIndex){

        // рассчитываем текущее положение верхнего левого элемента ядра на карте по индексу количества пройденных ядер свертки
        int minX = kernelIndex / (matrix.getColNum() - kernelSize.x + 1);
        int minY = kernelIndex % (matrix.getColNum() - kernelSize.y + 1);

        // рассчитываем положение текущего элемента в матрице
        int row =  curValue / matrix.getColNum();
        int col = curValue % matrix.getColNum();

        // возвращаем запрошенный элемент матрицы относительно его расположения в матрице ядра
        return matrix.getValue(minX + kernelX, minY + kernelY);
    }

    private double getValue(Matrix matrix, int curValue){
        int row =  curValue / matrix.getColNum();
        int col = curValue % matrix.getColNum();
        return matrix.getValue(row, col);
    }

    private Matrix calcLinksLayers(Layer x0, Layer y0, Layer x1, Layer y1, int batchsize, int indexMap) {
        int column = y0.getMapsSize().x * y0.getMapsSize().y;
        int row = y0.getKernelSize().x * y0.getKernelSize().y - 1 + column;
        Matrix differents = new Matrix(new Size(row, column));

        Size kernelSize = y0.getKernelSize();
        int x0Index = 0, y0Index = 0;

        boolean end = false;
        for (int j = 0; j < column; ) {
            for (int i = 0; i < row && !end; ) {

                // работаем с матрицей скрытого и входного слоев учитывая размерность матрицы весов
                for (int k = 0; k < kernelSize.x; k++) {
                    for (int l = 0; l < kernelSize.y; l++) {
                        double value = 0.0;
                        for (int m = 0; m < batchsize; m++) {
                            double x0Value = getValue(x0.getMap(m, indexMap), x0Index, kernelSize, k, l, j);
                            double y0Value = getValue(y0.getMap(m, indexMap), y0Index);
                            double x1Value = getValue(x1.getMap(m, indexMap), x0Index, kernelSize, k, l, j);
                            double y1Value = getValue(y1.getMap(m, indexMap), y0Index);
                            value += (x0Value * y0Value) - (x1Value * y1Value);
                        }
                        differents.setValue(i, j, value);
                        x0Index++;
                        i++;
                    }
                }
                y0Index++;
                j++;
                i = j;
                if (j == column){
                    end = true;
                }
            }
        }

        return differents;
    }
}
