package net;

public class ActivationFunction {
    public static double hyptan(double x){
        return (Math.pow(Math.E, x) - Math.pow(Math.E, -x)) / (Math.pow(Math.E, x) + Math.pow(Math.E, -x));
    }

    public static double sigm(double x){
        return 1 / (1 + Math.pow(Math.E, -x));
    }
}
