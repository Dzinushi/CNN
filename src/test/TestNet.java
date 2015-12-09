package test;

import java.io.IOException;
import dataset.Mnist;
import net.CNN;

public class TestNet {
    public static void main(String[] args) throws IOException {
        String imagesTest = "database/MNIST/test-images.idx3-ubyte";
        String labelsTest = "database/MNIST/test-labels.idx1-ubyte";
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelsTest, 10000);

        CNN cnn = new CNN();
        try {
            cnn = cnn.read("net_mnist_10000.cnn");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        cnn.test(testData);
        System.exit(0);
    }
}