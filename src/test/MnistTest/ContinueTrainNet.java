package test.MnistTest;

import dataset.Mnist;
import net.CNN;
import util.TaskToThread;

import java.io.IOException;

public class ContinueTrainNet {

    public static void main(String[] args) throws Exception {

        String imagesTrain = "database/MNIST/train-images.idx3-ubyte";
        String labelsTrain = "database/MNIST/train-labels.idx1-ubyte";
        String imagesTest = "database/MNIST/test-images.idx3-ubyte";
        String labelTest = "database/MNIST/test-labels.idx1-ubyte";

        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, labelsTrain, 60000);
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelTest, 10000);
        String netName = "";

        CNN cnn = new CNN();
        try {
            cnn = cnn.read(netName);
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        cnn.train(trainData, testData, 100);
        cnn.save(netName);

        TaskToThread.stop();
    }
}