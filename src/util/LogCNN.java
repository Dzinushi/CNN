package util;

public class LogCNN {
    public static void printInfo(final int iteration, final Precision precision, long time){
        double percent = (1.0 - (double)precision.getValue() / precision.getCount()) * 100;
        System.out.printf("%d) Precision: %d from %d; \t Errors: %.3f; \t Time train: %.3fs\n",
                iteration,
                precision.getValue(),
                precision.getCount(),
                percent,
                time / 1000.0);
    }

    public static void printAllTime(long time){
        System.out.printf("All time train: %.3fs", time / 1000.0);
    }
}
