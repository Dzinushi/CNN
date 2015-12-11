package test;

import dataset.Mnist;
import net.CNN;
import util.TaskToThread;

import java.io.IOException;

public class ContinueTrainNet {

    public static void main(String[] args) throws Exception {

        String imagesTrain = "database/MNIST/train-images.idx3-ubyte";
        String labelsTrain = "database/MNIST/train-labels.idx1-ubyte";
        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, labelsTrain, 60000);

        CNN cnn = new CNN();
        try {
            cnn = cnn.read("net_mnist_60000_new.cnn");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        cnn.train(trainData, 100);
        cnn.save("net_mnist_60000_new");

        TaskToThread.stop();
    }
}