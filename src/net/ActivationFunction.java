package net;

public class ActivationFunction {

    public enum function{
        HYPTAN,
        SIGM,
        RELU
    }

    public static double activation(function functionName, double x){
        switch (functionName){
            case HYPTAN:
                return hyptan(x);
            case SIGM:
                return sigm(x);
            case RELU:
                return relu(x);
        }
        return Double.NaN;
    }

    private static double hyptan(double x){
        return (Math.pow(Math.E, 2*x) - 1) / (Math.pow(Math.E, 2*x) + 1);
    }

    private static double sigm(double x){
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    private static double relu(double x){
        //return (x >= 0.0 ? x : 0.01*x);
        return (x >= 0.0 ? x : 0);
    }
}