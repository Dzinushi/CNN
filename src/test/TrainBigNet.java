package test;


import dataset.Mnist;
import net.CNN;
import net.CreateLayer;
import net.Layer;
import util.Size;
import util.TaskToThread;

import java.io.IOException;

public class TrainBigNet {
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
        trainData.load(imagesTrain, labelsTrain, 100);
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelTest, 100);

        CNN cnn = new CNN();
        cnn.setup(layers, 50);                  // batchsize
        cnn.setName(netName);
        cnn.autosave(false);
        cnn.setUsingAutoencoder(true);
        cnn.train(trainData, testData, 10);    // iterations

        TaskToThread.stop();
    }
}