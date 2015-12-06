package test;

import dataset.Mnist;
import net.CNN;

import java.io.IOException;

public class ContinueTrainNet {

    public static void main(String[] args) throws Exception {

        String imagesTrain = "database/MNIST/train-images.idx3-ubyte";
        String labelsTrain = "database/MNIST/train-labels.idx1-ubyte";
        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, labelsTrain, 60000);

        CNN cnn = new CNN();
        try {
            cnn = cnn.read("net_mnist.cnn");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        cnn.train(trainData, 1);
        cnn.save("net_mnist");
        System.exit(0);
    }
}