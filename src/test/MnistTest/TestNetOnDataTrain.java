package test.MnistTest;


import dataset.Mnist;
import net.CNN;
import util.LogCNN;
import util.Precision;
import util.TaskToThread;
import util.TimeCNN;

import java.io.IOException;

public class TestNetOnDataTrain {
    public static void main(String[] args) throws IOException {
        String imagesTest = "database/MNIST/train-images.idx3-ubyte";
        String labelsTest = "database/MNIST/train-labels.idx1-ubyte";
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelsTest, 60000);

        CNN cnn = new CNN();
        try {
            cnn = cnn.read("net_mnist_60000_relu");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        System.out.println("\nStart testing");
        TimeCNN timeTest = new TimeCNN();
        timeTest.start();

        Precision precision = cnn.test(testData);

        LogCNN.printTestInfo(precision, timeTest.getTimeLast());

        TaskToThread.stop();
    }
}
