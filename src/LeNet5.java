public class LeNet5 {
    double tanAmplitude = 1.7159;
    double tanOriginSlope = (2/3.0);


    private double nodeActivation(double input){
        //Take the sigmoid of the input and multiply it by the slope at origin
        double tanhInput = tanOriginSlope * (1/(1+Math.exp(-input)));
        //The activation function is a tanh function multiplied by a programmer-specified amplitude
        return tanAmplitude * ((Math.exp(tanhInput)-Math.exp(-tanhInput))/(Math.exp(tanhInput)+Math.exp(-tanhInput)));
    }

    private double derivative(double activation){
        //Based on the derivative of tanh: 1-(tanh^2)
        return 1 - activation * activation;
    }

    private double[][] convolve(double[][] matrix, double[][] filter, int start, int stride){
        return null;
    }
}
