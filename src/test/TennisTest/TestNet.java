package test.TennisTest;

import dataset.Tennis;
import net.CNN;
import util.LogCNN;
import util.Precision;
import util.TaskToThread;
import util.TimeCNN;

import java.io.IOException;

public class TestNet {
    public static void main(String[] args) throws IOException {
        String imagesTest = "database/TENNIS/tennis_train_data";
        String labelsTest = "database/TENNIS/tennis_train_label";

        Tennis testData = new Tennis();
        testData.load(imagesTest, labelsTest, 400);

        CNN cnn = new CNN();
        try {
            cnn = cnn.read("net_tennis_400");
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
