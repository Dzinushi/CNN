package test.MnistTest;

import dataset.Mnist;
import net.CNN;
import net.Layer;
import net.CreateLayer;
import util.Size;
import util.TaskToThread;

import java.io.IOException;

public class TrainNet {

    public static void main(String[] args) throws IOException {
        CreateLayer layers = new CreateLayer();
        layers.createLayer(Layer.inputLayer(new Size(28, 28)));
        layers.createLayer(Layer.convLayer(10, new Size(5, 5)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.convLayer(18, new Size(5, 5)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.outputLayer(10));

        String imagesTrain = "database/MNIST/train-images.idx3-ubyte";
        String labelsTrain = "database/MNIST/train-labels.idx1-ubyte";
        String imagesTest = "database/MNIST/test-images.idx3-ubyte";
        String labelTest = "database/MNIST/test-labels.idx1-ubyte";

        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, labelsTrain, 60000);
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelTest, 10000);

        CNN cnn = new CNN();
        cnn.setup(layers, 50);                  // batchsize
        cnn.setName("net_mnist_60000 (1-10-0.5-20-0.5-10)");
        cnn.autosave(true);
        cnn.train(trainData, testData, 100);    // iterations

        TaskToThread.stop();
    }
}