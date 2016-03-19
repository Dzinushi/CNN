package net;

import util.Matrix;
import util.MatrixOperation;
import util.Size;

class Autoencoder {

    private enum operation{
        ENCODE,
        DECODE
    }

    // layer        - сверточный слой
    // layerPrev    - входной / субдескритизирующий слой
    void start(Layer layer, Layer layerPrev, int indexMapOut, int epoch){

        train(layer, layerPrev, indexMapOut, operation.ENCODE);
        Layer layerOut = new Layer(layerPrev);
        train(layerOut, layer, indexMapOut, operation.DECODE);

        // изменение весов
        // разобраться с порогами. Пороги должны быть для каждого слоя, а веса для обоих слоев одинаковые
        for (int i = 0; i < epoch - 1; i++) {
            train(layerOut, layerPrev, indexMapOut, operation.ENCODE);
            train(layerPrev, layerOut, indexMapOut, operation.DECODE);
            // изменение весов
        }
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

    public void update(){

    }
}
