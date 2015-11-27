import dataset.Mnist;
import net.CNN;
import net.Layer;
import net.CreateLayer;
import util.Matrix;
import util.Size;
import util.Util;

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
        String lablesTrain = "database/MNIST/train-labels.idx1-ubyte";
        Mnist trainData = new Mnist();
        trainData.load(imagesTrain, lablesTrain, 100);

        CNN cnn = new CNN();
        cnn.setup(layers, 50);      // batchsize = 50
        cnn.train(trainData, 1);    // iterations = 1
    }
}
