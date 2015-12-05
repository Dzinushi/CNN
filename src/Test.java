import dataset.Mnist;
import net.CNN;
import net.Layer;
import net.CreateLayer;
import util.Size;

import java.io.IOException;

public class Test {

    public static void main(String[] args) throws IOException {
        CreateLayer layers = new CreateLayer();
        layers.createLayer(Layer.inputLayer(new Size(28, 28)));
        layers.createLayer(Layer.convLayer(6, new Size(5, 5)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.convLayer(12, new Size(5, 5)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.outputLayer(10));

        String imagesTrain = "database/MNIST/train-images.idx3-ubyte";
        String labelsTrain = "database/MNIST/train-labels.idx1-ubyte";
        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, labelsTrain, 60000);

        CNN cnn = new CNN();
        cnn.setup(layers, 50);      // batchsize
        cnn.train(trainData, 3);    // iterations

        String imagesTest = "database/MNIST/test-images.idx3-ubyte";
        String labelsTest = "database/MNIST/test-labels.idx1-ubyte";
        Mnist testData = new Mnist();
        testData.load(imagesTest, labelsTest, 60000);

        cnn.test(testData);

        System.exit(0);
    }
}