package util;

public class LogCNN {
    private static int iteration = 1;

    public static void printTrainInfo(final Precision precision, long time){
        double percent = (1.0 - (double)precision.getValue() / precision.getCount()) * 100;
        System.out.printf("%d) Train precision: %d from %d; \t Errors: %.3f; \t Time train: %.3fs\n",
                iteration,
                precision.getValue(),
                precision.getCount(),
                percent,
                time / 1000.0);
        ++iteration;
    }

    public static void printTestInfo(final Precision precision, long time){
        double percent = (1.0 - (double)precision.getValue() / precision.getCount()) * 100;
        System.out.printf("Test precision: %d from %d; \t Errors: %.3f; \t Time test: %.3fs\n",
                precision.getValue(),
                precision.getCount(),
                percent,
                time / 1000.0);
    }

    public static void printAllTime(long time){
        System.out.printf("\nAll time: %.3fs\n\n", time / 1000.0);
    }
}
