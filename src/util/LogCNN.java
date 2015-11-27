package util;

public class LogCNN {
    public static void printPrecision(final Precision precision){
        double percent = (double)precision.getValue() / precision.getCount();
        System.out.printf("Precision: %d from %d (%.3f)\n",
                precision.getValue(),
                precision.getCount(),
                percent);
    }
}
