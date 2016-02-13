package test;

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
        layers.createLayer(Layer.convLayer(8, new Size(5, 5)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.convLayer(16, new Size(5, 5)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.outputLayer(10));

        String imagesTrain = "database/MNIST/train-images.idx3-ubyte";
        String labelsTrain = "database/MNIST/train-labels.idx1-ubyte";
        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, labelsTrain, 60000);

        CNN cnn = new CNN();
        cnn.setup(layers, 20);      // batchsize
        cnn.train(trainData, 100);    // iterations
        cnn.save("net_mnist_60000_relu");

        TaskToThread.stop();
    }
}