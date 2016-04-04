package test.TennisTest;

import dataset.Tennis;
import net.CNN;
import util.TaskToThread;

import java.io.IOException;

public class ContinueTrain {
    public static void main(String[] args) throws Exception {

        String imagesTrain = "database/TENNIS/tennis_train_data";
        String labelsTrain = "database/TENNIS/tennis_train_label";
        String imagesTest = "database/TENNIS/tennis_train_data";
        String labelTest = "database/TENNIS/tennis_train_label";

        Tennis trainData = new Tennis();
        trainData.load(imagesTrain, labelsTrain, 400);
        Tennis testData = new Tennis();
        testData.load(imagesTest, labelTest, 400);
        String netName = "net_tennis_400_1";

        CNN cnn = new CNN();
        try {
            cnn = cnn.read(netName);
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        cnn.setAlpha(0.01);
        cnn.autosave(true);
        cnn.train(trainData, testData, 10000);

        TaskToThread.stop();
    }
}
