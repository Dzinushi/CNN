package test;

import java.io.IOException;
import dataset.Mnist;
import net.CNN;
import util.TaskToThread;

public class TestNet {
    public static void main(String[] args) throws IOException {
        String imagesTest = "database/MNIST/train-images.idx3-ubyte";
        String labelsTest = "database/MNIST/train-labels.idx1-ubyte";
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelsTest, 60000);

        CNN cnn = new CNN();
        try {
            cnn = cnn.read("net_mnist_60000.cnn");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        cnn.test(testData);

        TaskToThread.stop();
    }
}