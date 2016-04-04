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
        layers.createLayer(Layer.inputLayer(new Size(25, 15)));
        layers.createLayer(Layer.convLayer(4, new Size(6, 2)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.convLayer(8, new Size(5, 2)));
        layers.createLayer(Layer.sampLayer(new Size(2, 2)));
        layers.createLayer(Layer.outputLayer(11));

        String imagesTrain = "database/TENNIS/tennis_train_data";
        String labelsTrain = "database/TENNIS/tennis_train_label";
        String imagesTest = "database/TENNIS/tennis_test_data";
        String labelTest = "database/TENNIS/tennis_test_label";

        Tennis trainData = new Tennis();
        trainData.load(imagesTrain, labelsTrain, 400);
        Tennis testData = new Tennis();
        testData.load(imagesTest, labelTest, 400);

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
        cnn.setup(layers, 16);                          // batchsize
        cnn.setAlpha(0.8);
        cnn.setName("net_tennis_400_with_control_data");
        cnn.autosave(true);

        cnn.train(randTrainData, testData, 5000);       // iterations

        cnn.save("net_tennis_400_with_control_data");

        TaskToThread.stop();
    }
}
