package test.TennisTest;

import dataset.Tennis;
import net.CNN;
import net.CreateLayer;
import net.Layer;
import util.Size;
import util.TaskToThread;
import util.Util;

import java.io.IOException;

public class TrainNet {
    public static void main(String[] args) throws IOException {
        CreateLayer layers = new CreateLayer();
        layers.createLayer(Layer.inputLayer(new Size(30, 15)));
        layers.createLayer(Layer.convLayer(6, new Size(3, 2)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.convLayer(12, new Size(3, 2)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.outputLayer(4));

        String imagesTrain = "database/TENNIS/406 Forehands_data";
        String labelsTrain = "database/TENNIS/406 Forehands_label";
        String imagesTest = "database/TENNIS/406 Forehands_data";
        String labelTest = "database/TENNIS/406 Forehands_label";

        Tennis trainData = new Tennis();
        trainData.load(imagesTrain, labelsTrain, 60);
        Tennis testData = new Tennis();
        testData.load(imagesTest, labelTest, 60);

        // назначаем случайные индексы массиву
        int[] randIndex = Util.randPerm(trainData.getSize());

        Tennis randTrainData = new Tennis();
        randTrainData.setImageWidth(trainData.getImageWidth());
        randTrainData.setImageHeight(trainData.getImageHeight());
        randTrainData.setSize(trainData.getSize());

        for (int i = 0; i < randIndex.length; i++) {
            randTrainData.setData(i, trainData.getData(randIndex[i]));
            randTrainData.setLabel(i, trainData.getLabel(randIndex[i]));
        }

        CNN cnn = new CNN();
        cnn.setup(layers, 12);                  // batchsize
        cnn.setName("net_mnist_60 (1-10-0.5-20-0.5-10)");
        cnn.autosave(true);
        cnn.train(randTrainData, testData, 100);    // iterations

        TaskToThread.stop();
    }
}
