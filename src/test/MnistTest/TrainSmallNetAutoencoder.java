package test.MnistTest;


import dataset.Mnist;
import net.CNN;
import net.CreateLayer;
import net.Layer;
import util.Size;
import util.TaskToThread;

import java.io.IOException;

public class TrainSmallNetAutoencoder {
    public static void main(String[] args) throws IOException {
        CreateLayer layers = new CreateLayer();
        layers.createLayer(Layer.inputLayer(new Size(28, 28)));
        layers.createLayer(Layer.convLayer(2, new Size(5, 3)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.convLayer(4, new Size(5, 2)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.outputLayer(10));

        String imagesTrain = "database/MNIST/train-images.idx3-ubyte";
        String labelsTrain = "database/MNIST/train-labels.idx1-ubyte";
        String imagesTest = "database/MNIST/test-images.idx3-ubyte";
        String labelTest = "database/MNIST/test-labels.idx1-ubyte";

        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, labelsTrain, 177);
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelTest, 177);

        CNN cnn = new CNN();
        cnn.setName("cnn_1000_2_(0.5)_4_(0.5)_10_autoencoder");
        cnn.setup(layers, 10);                  // batchsize
        cnn.autosave(false);
        cnn.setUsingAutoencoder(false);
        cnn.train(trainData, testData, 100);    // iterations

        TaskToThread.stop();
    }
}