package net;

import dataset.Mnist;
import util.Matrix;
import util.MatrixOperation;
import util.Size;
import util.TaskToThread;

import java.io.IOException;

public class Autoencoder {

    public static void main(String[] args) throws IOException {
        CreateLayer layers = new CreateLayer();
        layers.createLayer(Layer.inputLayer(new Size(28, 28)));
        layers.createLayer(Layer.convLayer(4, new Size(5, 5)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.convLayer(8, new Size(5, 5)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.outputLayer(10));

        String imagesTrain = "database/MNIST/train-images.idx3-ubyte";
        String labelsTrain = "database/MNIST/train-labels.idx1-ubyte";
        String imagesTest = "database/MNIST/test-images.idx3-ubyte";
        String labelTest = "database/MNIST/test-labels.idx1-ubyte";

        String netName = "test_relu_cnn";

        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, labelsTrain, 10000);
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelTest, 10000);

        CNN cnn = new CNN();
        cnn.setup(layers, 50);                  // batchsize
        cnn.setName(netName);
        cnn.autosave(false);
        cnn.train(trainData, testData, 10);    // iterations

        TaskToThread.stop();
    }

    private void train(Layer layer, Layer layerPrev, int indexMapOut){
        TaskToThread taskToThread = new TaskToThread(layer.getMapOutNumber()) {
            @Override
            public void start(int start, int end) {
                for (int i = start; i < end; i++) {
                    Matrix s = null;
                    for (int j = 0; j < layerPrev.getMapOutNumber(); j++) {
                        if (s != null){
                            Matrix sCur = encode(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i));
                            s = MatrixOperation.operation(s, sCur, MatrixOperation.Op.SUM);
                        }
                        else {
                            s = encode(layerPrev.getMap(indexMapOut, j), layer.getKernel(j,i));
                        }
                    }
                    s = activation(s, layer.getT(i), ActivationFunction.function.SIGM);
                    layer.setMapOutValue(indexMapOut, i, s);
                }
            }
        };
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
}
