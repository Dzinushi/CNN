package test.MnistTest;

import java.io.IOException;
import dataset.Mnist;
import net.CNN;
import util.LogCNN;
import util.Precision;
import util.TaskToThread;
import util.TimeCNN;

public class TestNetOnDataTest {
    public static void main(String[] args) throws IOException {
        String imagesTest = "database/MNIST/test-images.idx3-ubyte";
        String labelsTest = "database/MNIST/test-labels.idx1-ubyte";
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelsTest, 10000);

        CNN cnn = new CNN();
        try {
            cnn = cnn.read("net_mnist_60000");
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