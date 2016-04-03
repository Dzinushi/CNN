package net;

import util.Matrix;
import util.MatrixOperation;
import util.Size;
import util.TaskToThread;

import java.util.ArrayList;
import java.util.List;

class Autoencoder {

    private List<Layer> inputs;   // список весовых коэффициентов и номер слоя, которому они принадлежат
    private List<Layer> hiddens;   // список порогов и номер слоя, которому они принадлежат

    public Autoencoder(){
        inputs = new ArrayList<>();
        hiddens = new ArrayList<>();
    }

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
    void start(Layer hidden, Layer input, int batchsize, int indexLayer){

        Layer out, hidden1;

        if (inputs.size() <= indexLayer) {
            out = new Layer(input);
            out.setTSize();                 // входной слой использует свои пороговые значения, количество которых равнятется числу карт

            hidden1 = new Layer(hidden);

            inputs.add(out);
            hiddens.add(hidden1);
        }
        else {
            out = inputs.get(indexLayer);
            hidden1 = hiddens.get(indexLayer);
        }

        for (int i = 0; i < batchsize; i++) {
            train(out, hidden, i, operation.DECODE);
            train(hidden1, out, i, operation.ENCODE);
        }

        // обновление ядра / ядер свертки на сверточном слое hidden
        update(input, hidden, out, hidden1, batchsize);
    }

    private void train(Layer layer, Layer layerPrev, int indexMapOut, operation operation){
        TaskToThread taskToThread = new TaskToThread(layer.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int i = start; i < end; i++) {
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
                    s = activation(s, layer.getT(i), ActivationFunction.function.SIGM);
                    layer.setMapOutValue(indexMapOut, i, s);
                }
            }
        };
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

    // изменить обновление порогов: сделать i и j пороги для x(0) - x(1) и y(0) - y(1)
    private void update(Layer x0, Layer y0, Layer x1, Layer y1, int batchsize){
        updateKernel(x0, y0, x1, y1, batchsize);
        updateT(x0, x1, batchsize, 1);
        updateT(y0, y1, batchsize, 0);
    }

    private void updateKernel(Layer x0, Layer y0, Layer x1, Layer y1, int batchsize){
        // обновление весов
        // получение списка весовых коэффициентов для данного изображения
        // i - номер элемента групповой выбрки
        // j - номер карты элемента группповой выборки
        TaskToThread taskToThread = new TaskToThread(x0.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int n = start; n < end; n++) {
                    for (int i = 0; i < y0.getKernel().size(); i++) {

                        List<Matrix> kernels = y0.getKernel().get(i);
                        for (int j = 0; j < kernels.size(); ++j) {

                            // формируем матрицу связей между входным и скрытым слоем, в которой число строк равно числу элементов входного слоя, а число столбцов числу элементов скрытого слоя
                            Matrix differences = calcLinksLayers(x0, y0, x1, y1, batchsize, n, j);

                            Matrix kernel = kernels.get(j);
                            int num = 0;
                            for (int k = 0; k < kernel.getColNum(); k++) {
                                for (int l = 0; l < kernel.getRowNum(); l++) {
                                    double value = sumSimilarWeights(differences, num) / differences.getColNum();
                                    kernel.setValue(l, k, kernel.getValue(k,l) + ((0.5 / batchsize) * value));
                                    num++;
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    private void updateT(Layer come, Layer recover, int batchsize, int saveResultIndexLayer){
        for (int i = 0; i < come.getMapOutNumber(); i++) {
            double[] differences = calcLinksLayers(come, recover, batchsize, i);
            double value = 0.0;
            for (double difference : differences) {
                value += difference;
            }
            value /= differences.length;
            if (saveResultIndexLayer == 0) {
                come.setTValue(i, come.getT(i) + (0.5 / batchsize) * value);
            }
            else {
                recover.setTValue(i, recover.getT(i) + (0.5 / batchsize) * value);
            }
        }
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
        int curX = kernelIndex / (matrix.getColNum() - kernelSize.x + 1);
        int curY = kernelIndex % (matrix.getColNum() - kernelSize.y + 1);

        // возвращаем запрошенный элемент матрицы относительно его расположения в матрице ядра
        return matrix.getValue(curX + kernelX, curY + kernelY);
    }

    private double getValue(Matrix matrix, int curValue){
        int row =  curValue / matrix.getColNum();
        int col = curValue % matrix.getColNum();
        return matrix.getValue(row, col);
    }

    private double[] calcLinksLayers(Layer come, Layer recover, int batchsize, int indexMap){
        int size = come.getMapsSize().x * come.getMapsSize().y;   // recover и come ВСЕГДА имеют одинаковый размер
        double[] differences = new double[size];

        int index = 0;
        for (int i = 0; i < differences.length; i++) {
            double value = 0.0;
            for (int m = 0; m < batchsize; m++) {
                double comeValue = getValue(come.getMap(m, indexMap), index);
                double recValue = getValue(recover.getMap(m, indexMap), index);
                value += comeValue - recValue;
            }
            differences[i] = value / batchsize;
            index++;
        }

        return differences;
    }

    private Matrix calcLinksLayers(Layer x0, Layer y0, Layer x1, Layer y1, int batchsize, int indexInputMap, int indexHiddenMap) {
        int column = y0.getMapsSize().x * y0.getMapsSize().y;
        int row = y0.getKernelSize().x * y0.getKernelSize().y - 1 + column;
        Matrix differences = new Matrix(new Size(row, column));

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
                            double x0Value = getValue(x0.getMap(m, indexInputMap), x0Index, kernelSize, k, l, j);
                            double y0Value = getValue(y0.getMap(m, indexHiddenMap), y0Index);
                            double x1Value = getValue(x1.getMap(m, indexInputMap), x0Index, kernelSize, k, l, j);
                            double y1Value = getValue(y1.getMap(m, indexHiddenMap), y0Index);
                            value += (x0Value * y0Value) - (x1Value * y1Value);
                        }
                        differences.setValue(i, j, value / batchsize);
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

        return differences;
    }
}
