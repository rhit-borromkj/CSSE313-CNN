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

    /**
     * Calculates a single output of the convolution
     * @param matrix - the image to be filtered
     * @param filter - the filter kernel
     * @param x - the x position of the pixel
     * @param y - the y position of the pixel
     * @return - the double new pixel value of the convolution
     */
    private double convolvePixel(double[][] matrix, double[][] filter, int x, int y){
        double output = 0;
        for(int i = 0; i < filter.length; i++){
            for(int j = 0; j < filter[i].length; j++){
                output += (matrix[x+i][y+j]*filter[i][j]);
            }
        }
        return output;
    }

    /**
     * Convolves a 2D matrix with a 2D filter without stride or padding
     * @param matrix - the 2D matrix representing an image
     * @param filter - the filter kernel
     * @return - the double[][] array of the new image after convolution
     */
    private double[][] convolve(double[][] matrix, double[][] filter){
        int resultWidth = matrix.length - filter.length + 1;
        int resultHeight = matrix[0].length - filter[0].length + 1;
        double[][] output = new double[resultWidth][resultHeight];

        for(int i = 0; i < resultWidth; i++){
            for(int j = 0; j < resultHeight; j++){
               output[i][j] = convolvePixel(matrix, filter, i, j);
            }
        }
        return output;
    }

    /**
     * Convolves a 2D matrix with a 2D filter with padding
     * @param matrix - the 2D matrix representing an image
     * @param filter - the filter kernel
     * @return - the double[][] array of the new image after convolution
     */
    private double[][] convolvePadded(double[][] matrix, double[][] filter, int outputWidth, int outputHeight){
        int horizontalPadding = (outputWidth - matrix.length + filter.length - 1)/2;
        int verticalPadding = (outputHeight - matrix[0].length + filter[0].length - 1)/2;
        double[][] output = new double[outputWidth][outputHeight];

        //Pad the matrix
        double[][] paddedMatrix = new double[outputWidth + horizontalPadding][outputHeight + verticalPadding];
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[i].length; j++){
                paddedMatrix[i + horizontalPadding][j + verticalPadding] = matrix[i][j];
            }
        }

        //Convolve the matrix
        for(int i = 0; i < outputWidth + horizontalPadding - filter.length + 1; i++){
            for(int j = 0; j < outputHeight + horizontalPadding - filter[0].length + 1; j++){
                output[i][j] = convolvePixel(paddedMatrix, filter, i, j);
            }
        }
        return output;
    }

    /**
     * Convolves a 2D matrix with a 2D filter with a stride greater than 1
     * @param matrix - the 2D matrix representing an image
     * @param filter - the filter kernel
     * @param horizontalStride - the horizontal stride of the convolution
     * @param verticalStride - the vertical stride of the convolution
     * @return - the double[][] array of the new image after convolution
     */
    private double[][] convolveStrided(double[][] matrix, double[][] filter, int horizontalStride, int verticalStride){
        int resultWidth = (matrix.length - filter.length + horizontalStride)/horizontalStride;
        int resultHeight = (matrix[0].length - filter[0].length + verticalStride)/verticalStride;
        double[][] output = new double[resultWidth][resultHeight];

        for(int i = 0; i < resultWidth; i++){
            for(int j = 0; j < resultHeight; j++){
                output[i][j] = convolvePixel(matrix, filter, i*horizontalStride, j*verticalStride);
            }
        }
        return output;
    }

    /**
     * Puts a set of matrices through a pooling function by summing the values of the matrix inside of the
     * pooling grid, then multiplying by a trainable weight and adding a trainable bias, then using a
     * sigmoid activation on the outcome. Used for the pooling layers of the CNN
     * @param matrices - the set of matrices to be pooled
     * @param poolingWeights - the trainable weights the sums are multiplied by
     * @param biases - the trainable biases added to the multiplied sums
     * @param poolWidth - the width of the pooling grid
     * @param poolHeight - the height of the pooling grid
     * @param stride - how many units over the pooling grid will move between iterations
     * @return - the double[][][] set of pooled matrices
     */
    private double[][][] pool(double[][][] matrices, double[] poolingWeights, double[] biases, int poolWidth, int poolHeight, int stride){
        int outputWidth = matrices[0].length / poolWidth;
        int outputHeight = matrices[0][0].length / poolHeight;
        double[][][] output = new double[matrices.length][outputWidth][outputHeight];

        for(int m = 0; m < matrices.length; m++){
            for(int i = 0; i < (matrices[m].length - poolWidth + stride)/stride; i++){
                for(int j = 0; j < (matrices[m][i].length - poolHeight + stride)/stride; j++){
                    //Sum all of the values in the embossed area on the matrix
                    for(int k = 0; k < poolWidth; k++){
                        for(int l = 0; l < poolHeight; l++){
                            output[m][i][j] += matrices[m][i*stride+k][j*stride+l];
                        }
                    }
                    //Multiply the sum by the pooled weight at that point
                    output[m][i][j] = output[m][i][j] * poolingWeights[m] + biases[m];
                }
            }
        }

        return output;
    }

    /**
     * Tests the network's convolution functionality
     */
    public void testConvolution(){
        double[][] matrix = { {2, 3, 7, 4, 6, 2, 9},
                              {6, 6, 9, 8, 7, 4, 3},
                              {3, 4, 8, 3, 8, 9, 7},
                              {7, 8, 3, 6, 6, 3, 4},
                              {4, 2, 1, 8, 3, 4, 6},
                              {3, 2, 4, 1, 9, 8, 3},
                              {0, 1, 3, 9, 2, 1, 4}
                             };

        double[][] filter = { {3, 4, 4},
                              {1, 0, 2},
                              {-1, 0, 3}
                            };

        double[][] convolvedMatrix = convolveStrided(matrix, filter, 2, 2);
        for(int i = 0; i < convolvedMatrix.length; i++){
            for(int j = 0; j < convolvedMatrix[i].length; j++){
                System.out.print(convolvedMatrix[i][j] + " ");
            }
            System.out.println();
        }

        System.out.println("\nDesired Output:");
        System.out.println("91.0  100.0  83.0");
        System.out.println("69.0  91.0  127.0");
        System.out.println("44.0  72.0   74.0");
    }

    /**
     * Tests the network's pooling functionality
     */
    public void testPooling(){
        double[][][] matrices = { {{2, 2, 7, 3},
                {9, 4, 6, 1},
                {8, 5, 2, 4},
                {3, 1, 2, 6}}
        };

        double[] poolingWeights = {1, 1, 1, 1, 1, 1};
        double[] biases = {1, 1, 1, 1, 1, 1};

        double[][][] pooledMatrices = pool(matrices, poolingWeights, biases, 2, 2, 2);

        System.out.println("Output: ");
        for(double[][] matrix : pooledMatrices){
            for(int r = 0; r < matrix.length; r++){
                for(int c = 0; c < matrix[0].length; c++){
                    System.out.print(matrix[r][c] + " ");
                }
                System.out.println();
            }
        }

        System.out.println("\nDesired Output:");
        System.out.println("17.0  17.0");
        System.out.println("17.0  14.0");
    }

}
