import javax.swing.*;

public class LeNet5 {

    //Programmer-defined constants
    private final double tanAmplitude = 1.7159;
    private final double tanOriginSlope = (2/3.0);
    private double trainingSetSize = 0;
    private double learningRate = 0.1;
    private int inputSize = 28*28; 	// Fixed for now.
    private double inputs[][][];
    private double desiredOutputs[]; // Single desired output
    private final int filterWidth = 5; // Filter width for all convolution layers
    private final int filterHeight = 5; // Filter height for all convolution layers
    private final int c1Size = 6;
    private final int c1Width = 28;
    private final int c1Height = 28;
    private final int s2Size = 6;
    private final int s2Width = 14;
    private final int s2Height = 14;
    private final int c3Size = 16;
    private final int c3Width = 10;
    private final int c3Height = 10;
    private final int s4Size = 16;
    private final int c5Size = 120;
    private final int f6Size = 84;
    private final int outputSize = 10;

    //Weight Matrices
    private double[][][] c1Filters;
    private double[] c1Biases;
    private double[] s2Weights;
    private double[] s2Biases;
    private double[][][] c3Filters;
    private double[] c3Biases;
    private double[] s4Weights;
    private double[] s4Biases;
    private double[][][] c5Filters;
    private double[] c5Weights;
    private double[] c5Biases;
    private double[][] f6Weights;
    private double[] f6Biases;
    private double[][] outputWeights;

    /**
     * Initializes the network on the given inputs and desired outputs
     * @param inputs - double[][][]
     * @param desiredOutputs - double[]
     */
    public void initNetwork(double[][][] inputs, double[] desiredOutputs) {
        //Initialize the inputs and desired outputs
        this.inputs = inputs;
        this.trainingSetSize = inputs.length;
        if (this.trainingSetSize == 0) {
            System.out.println("No training data.");
            System.exit(0);
        }
        this.desiredOutputs = desiredOutputs;

        //Initialize all of the network's trainable parameters
        this.c1Filters = new double[c1Size][filterWidth][filterHeight];
        this.c1Biases = new double[c1Size];
        this.s2Weights = new double[s2Size];
        this.s2Biases = new double[s2Size];
        this.c3Filters = new double[c3Size][filterWidth][filterHeight];
        this.c3Biases = new double[c3Size];
        this.s4Weights = new double[s4Size];
        this.s4Biases = new double[s4Size];
        this.c5Filters = new double[c5Size][filterWidth][filterHeight];
        this.c5Weights = new double[c5Size];
        this.c5Biases = new double[c5Size];
        this.f6Weights = new double[f6Size][c5Size];
        this.f6Biases = new double[f6Size];
        this.outputWeights = new double[outputSize][f6Size];

        //Initialize weights
//        initializeWeights(this.c1Filters, inputSize);
//        initializeWeights(this.c1Biases, inputSize);
//        initializeWeights(this.s2Weights, c1Width*c1Height);
//        initializeWeights(this.s2Biases, c1Size);
//        initializeWeights(this.c3Filters, s2Width*s2Height);
//        initializeWeights(this.c3Biases, s2Size);
//        initializeWeights(this.s4Weights, c3Width*c3Height);
//        initializeWeights(this.s4Biases, c3Size);
//        initializeWeights(this.c5Filters, s4Width*s4Height);
//        initializeWeights(this.c5Weights, s4Size);
//        initializeWeights(this.c5Biases, s4Size);
//        initializeWeights(this.f6Weights, c5Size);
//        initializeWeights(this.f6Biases, c5Size);
//        initializeOutputWeights(this.outputWeights);
    }

    /**
     * Does a simple test of the network using set weights and no activation functions
     * to demonstrate the functionality. To make things easier to call, the
     * displayImage methods are in this class, but these will be removed
     * for Milestone 2
     */
    public void testNetworkSimple(){
        initializeTestWeightsSimple();
        double[] binaryDesiredOutput = binaryEncodeSolution(desiredOutputs[0]);
        displayImage(inputs[0], 28, "Input");

        //C1
        double[][][] c1 = new double[c1Size][c1Width][c1Height];
        for(int f = 0; f < c1Filters.length; f++){
            c1[f] = convolvePadded(inputs[0], c1Filters[f], c1Biases[f], c1Width, c1Height);
        }
        displayImage(c1[0], 28, "C1");

        //S2
        double[][][] s2 = new double[s2Size][s2Width][s2Height];
        s2 = pool(c1, s2Weights, s2Biases, 2, 2, 2);
        displayImage(s2[0], 14, "S2");

        //C3
        double[][][] c3 = new double[c3Size][c3Width][c3Height];
        //First 6 filters (0..5)
        for(int f = 0; f < 6; f++){
            for(int i = 0; i < c3Width; i++){
                for(int j = 0; j < c3Height; j++){
                    c3[f][i][j] = convolvePixel(s2[f%s2Size], c3Filters[f], i, j)
                            + convolvePixel(s2[(f+1)%s2Size], c3Filters[f], i, j)
                            + convolvePixel(s2[(f+2)%s2Size], c3Filters[f], i, j)
                            + c3Biases[f];
                }
            }
        }
        //Next 9 Filters (6..11)
        for(int f = 6; f < 12; f++){
            for(int i = 0; i < c3Width; i++){
                for(int j = 0; j < c3Height; j++){
                    c3[f][i][j] = convolvePixel(s2[(f-6)%s2Size], c3Filters[f], i, j)
                            + convolvePixel(s2[(f-5)%s2Size], c3Filters[f], i, j)
                            + convolvePixel(s2[(f-4)%s2Size], c3Filters[f], i, j)
                            + convolvePixel(s2[(f-3)%s2Size], c3Filters[f], i, j)
                            + c3Biases[f];
                }
            }
        }
        //Next 3 Filters (12..14)
        for(int f = 12; f < 15; f++){
            for(int i = 0; i < c3Width; i++){
                for(int j = 0; j < c3Height; j++){
                    c3[f][i][j] = convolvePixel(s2[(f-12)%s2Size], c3Filters[f], i, j)
                            + convolvePixel(s2[(f-11)%s2Size], c3Filters[f], i, j)
                            + convolvePixel(s2[(f-9)%s2Size], c3Filters[f], i, j)
                            + convolvePixel(s2[(f-8)%s2Size], c3Filters[f], i, j)
                            + c3Biases[f];
                }
            }
        }
        //Last Filter (15)
        for(int i = 0; i < c3Width; i++){
            for(int j = 0; j < c3Height; j++){
                for(int f = 0; f < 6; f++){
                    c3[15][i][j] += convolvePixel(s2[f], c3Filters[15], i, j);
                }
            }
        }
        displayImage(c3[0], 10, "C3");

        //S4: Pooling of the 16 matrices from C3 into 16 5x5 matrices using 2x2 filters with 2-bit stride
        double[][][] s4 = pool(c3, s4Weights, s4Biases, 2, 2, 2);
        displayImage(s4[0], 5, "S4");

        //C5
        double[] c5 = new double[c5Size];
        for(int f = 0; f < c5Size; f++){
            for(int s = 0; s < s4Size; s++){
                c5[f] += convolvePixel(s4[s], c5Filters[f], 0, 0);
            }
            c5[f] = c5[f] * c5Weights[f] + c5Biases[f];
        }
        displayImage(c5, 120, "C5");

        //F6
        double[] f6 = new double[f6Size];
        for(int h = 0; h < f6Size; h++){
            for(int n = 0; n < c5Size; n++){
                f6[h] += c5[n] * f6Weights[h][n];
            }
            f6[h] += f6Biases[h];
        }
        displayImage(f6, 84, "F6");

        //Output
        double[] output = new double[outputSize];
        for(int o = 0; o < outputSize; o++){
            for(int h = 0; h < f6Size; h++){
                output[o] += f6[h] * outputWeights[o][h];
            }
        }
        displayImage(output, 10, "Output");
    }

    /**
     * Trains the network the epoch number of times using Convolutional artificial
     * intelligence methods. There are 8 layers in the network: input layer, 3 convolutional
     * layers, 2 pooling layers, a feed-forward hidden layer, then an output layer. The
     * convolutional and pooling layers work to filter the inputted image so it can
     * be read by the feed-forward classifier portion of the network
     * Training is SLOW
     * @param epochs - int
     */
    public void trainNetwork(int epochs){
        for(int e = 0; e < epochs; e++){
            for (int t = 0; t < trainingSetSize; t++) {
                //TODO: Implement tanh activation on all layers up to F6

                //Translate the desired output digit into a binary-encoded array
                double[] binaryDesiredOutput = binaryEncodeSolution(desiredOutputs[t]);

                //C1: Padded convolution of the input from a 32 x 32 to 28 x 28 using 5 x 5 filter with 1-bit stride
                double[][][] c1 = new double[c1Size][c1Width][c1Height];
                for(int f = 0; f < c1Filters.length; f++){
                    c1[f] = convolvePadded(inputs[t], c1Filters[f], c1Biases[f], c1Width, c1Height);
                }

                //S2: Pooling of the 6 matrices form C1 into 14x14 matrices using 6 2x2 filters with 2-bit stride
                double[][][] s2 = pool(c1, s2Weights, s2Biases, 2, 2, 2);

                //C3: Convolution of the 6 matrices in S2 into 16 10x10 matrices using 16 5x5 filters with 2-bit stride
                /*
                 *   0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
                 * 0 X       X X X     X X  X  X     X  X
                 * 1 X X       X X X     X  X  X  X     X
                 * 2 X X X       X X X      X     X  X  X
                 * 3   X X X     X X X X       X     X  X
                 * 4     X X X     X X X X     X  X     X
                 * 5       X X X     X X X  X     X  X  X
                 */
                double[][][] c3 = new double[c3Size][c3Width][c3Height];
                //First 6 filters (0..5)
                for(int f = 0; f < 6; f++){
                    for(int i = 0; i < c3Width; i++){
                        for(int j = 0; j < c3Height; j++){
                            c3[f][i][j] = convolvePixel(s2[f%s2Size], c3Filters[f], i, j)
                                    + convolvePixel(s2[(f+1)%s2Size], c3Filters[f], i, j)
                                    + convolvePixel(s2[(f+2)%s2Size], c3Filters[f], i, j)
                                    + c3Biases[f];
                        }
                    }
                }
                //Next 9 Filters (6..11)
                for(int f = 6; f < 12; f++){
                    for(int i = 0; i < c3Width; i++){
                        for(int j = 0; j < c3Height; j++){
                            c3[f][i][j] = convolvePixel(s2[(f-6)%s2Size], c3Filters[f], i, j)
                                    + convolvePixel(s2[(f-5)%s2Size], c3Filters[f], i, j)
                                    + convolvePixel(s2[(f-4)%s2Size], c3Filters[f], i, j)
                                    + convolvePixel(s2[(f-3)%s2Size], c3Filters[f], i, j)
                                    + c3Biases[f];
                        }
                    }
                }
                //Next 3 Filters (12..14)
                for(int f = 12; f < 15; f++){
                    for(int i = 0; i < c3Width; i++){
                        for(int j = 0; j < c3Height; j++){
                            c3[f][i][j] = convolvePixel(s2[(f-12)%s2Size], c3Filters[f], i, j)
                                    + convolvePixel(s2[(f-11)%s2Size], c3Filters[f], i, j)
                                    + convolvePixel(s2[(f-9)%s2Size], c3Filters[f], i, j)
                                    + convolvePixel(s2[(f-8)%s2Size], c3Filters[f], i, j)
                                    + c3Biases[f];
                        }
                    }
                }
                //Last Filter (15)
                for(int i = 0; i < c3Width; i++){
                    for(int j = 0; j < c3Height; j++){
                        for(int f = 0; f < 6; f++){
                            c3[15][i][j] += convolvePixel(s2[f], c3Filters[15], i, j);
                        }
                    }
                }

                //S4: Pooling of the 16 matrices from C3 into 16 5x5 matrices using 2x2 filters with 2-bit stride
                double[][][] s4 = pool(c3, s4Weights, s4Biases, 2, 2, 2);

                //C5: Convolution of 16 matrices from S4 into a single 120-node array using 16 5x5 filters and 120 additional weights
                double[] c5 = new double[c5Size];
                for(int f = 0; f < c5Size; f++){
                    for(int s = 0; s < s4Size; s++){
                        c5[f] += convolvePixel(s4[s], c5Filters[f], 0, 0);
                    }
                    c5[f] = c5[f] * c5Weights[f] + c5Biases[f];
                }

                //F6: Feed-forward fully connected hidden layer with 120 inputs, 84 hidden weights per input, and 84 biases
                double[] f6 = new double[f6Size];
                for(int h = 0; h < f6Size; h++){
                    for(int n = 0; n < c5Size; n++){
                        f6[h] += c5[n] * f6Weights[h][n];
                    }
                    f6[h] += f6Biases[h];
                }

                //Output: Feed-forward fully connected output layer with 84 inputs, 10 weights per input
                double[] output = new double[outputSize];
                for(int o = 0; o < outputSize; o++){
                    for(int h = 0; h < f6Size; h++){
                        output[o] += f6[h] * outputWeights[o][h];
                    }
                }
            }
            System.out.println("Epoch " + (e+1) + " completed.");
        }

        System.out.println("\nDone training.");
    }

    /**
     * The CNNet's activation function according to the paper found at
     * <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791">
     *     Gradient-Based Learning Applied to Document Recognition by Yann LeCun et al
     * </a>
     * @param input - the input to the node
     * @return - the activation of the node
     */
    private double tanh(double input){
        //Take the sigmoid of the input and multiply it by the slope at origin
        double tanhInput = tanOriginSlope * sigmoid(input);
        //The activation function is a tanh function multiplied by a programmer-specified amplitude
        return tanAmplitude * ((Math.exp(tanhInput)-Math.exp(-tanhInput))/(Math.exp(tanhInput)+Math.exp(-tanhInput)));
    }

    /**
     * A sigmoid activation function
     * @param input - the input
     * @return - the double sigmoid activation value
     */
    private double sigmoid(double input){
        return 1/(1+Math.exp(-input));
    }

    /**
     * Activation function of the output layer
     * @param input - all of the ACTIVATED INPUTS that go into the output layer
     * @return - the output of this output node
     */
    private double outputActivation(int output, double[] input){
        double activation = 0;
        for(int i = 0; i < input.length; i++){
            activation += Math.pow((input[i] - outputWeights[output][i]),2);
        }
        return activation;
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
     * Convolves a 2D matrix with a 2D filter with padding
     * @param matrix - the 2D matrix representing an image
     * @param filter - the filter kernel
     * @param bias - the trainable bias associated with the convolution filter
     * @param outputWidth - the resulting width of the output
     * @param outputHeight - the resulting height of the output
     * @return - the double[][] array of the new image after convolution
     */
    private double[][] convolvePadded(double[][] matrix, double[][] filter, double bias, int outputWidth, int outputHeight){
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
                //TODO: Implement tanh activation on the convolved pixel
                output[i][j] = convolvePixel(paddedMatrix, filter, i, j) + bias;
            }
        }
        return output;
    }

    /**
     * Puts a set of matrices through an average pooling function by summing the values of the matrix inside the
     * pooling grid, then taking the average and multiplying that by a trainable weight and adding a trainable bias,
     * Used for the pooling layers of the CNN
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
                    //Multiply the average by the pooled weight at that point and add biases
                    //TODO: Sigmoid??? The activation
                    output[m][i][j] = (output[m][i][j]/(poolWidth*poolHeight)) * poolingWeights[m] + biases[m];
                }
            }
        }

        return output;
    }

    /**
     * Initializes the weights of the given 3D array to a uniform distribution
     * between -2.4/Fi to 2.4/Fi, where Fi is the size of the input TO the layer using
     * this weight matrix
     * @param weights - the weight matrix to be initialized
     * @param Fi - the size of the input to the layer
     */
    public void initializeWeights(double[][][] weights, int Fi){
        for(int i = 0; i < weights.length; i++){
            for(int j = 0; j < weights[i].length; j++){
                for(int k = 0; k < weights[i][j].length; k++){
                    weights[i][j][k] = Math.random()*(2.4/Fi + 1 - (-2.4/Fi) + 2.4/Fi);
                }
            }
        }
    }

    /**
     * Initializes the weights of the given 1D array to a uniform distribution
     * between -2.4/Fi to 2.4/Fi, where Fi is the size of the input TO the layer using
     * this weight matrix
     * @param weights - the weight array to be initialized
     * @param Fi - the size of the input to the layer
     */
    public void initializeWeights(double[] weights, int Fi){
        for(int i = 0; i < weights.length; i++){
            weights[i] = Math.random()*(2.4/Fi + 1 - (-2.4/Fi) + 2.4/Fi);
        }
    }

    /**
     * Initializes the weights of the given 2D array to a uniform distribution
     * of either -1 or 1 (USED TO INITIALIZE OUTPUT WEIGHTS)
     * @param outputWeights - the weight array to be initialized
     */
    public void initializeOutputWeights(double[][] outputWeights){
        for(int i = 0; i < outputWeights.length; i++){
            for(int j = 0; j < outputWeights[i].length; j++){
                outputWeights[i][j] = Math.random() > 0.5 ? 1 : -1;
            }
        }
    }

    /**
     * Converts the solution parameter into a binary encoding, where the value at the
     * solution parameter index is 1, and the rest of the encoding is all 0.
     *
     * For example: The encoding for 5 is {0,0,0,0,0,1,0,0,0,0}
     * @param solution - double (gets cast to an int)
     * @return double[] the binary encoding
     */
    public double[] binaryEncodeSolution(double solution){
        double[] binaryDesiredOutput = new double[outputSize];
        int solutionLocation = (int)solution;
        binaryDesiredOutput[solutionLocation] = 1;
        return binaryDesiredOutput;
    }




    /**
     * Uses the PixelGrid class to display the 1D image read from the
     * MNIST data set
     * @param image - int[]
     */
    public static void displayImage(double[]image, int size, String name) {
        JFrame window = new JFrame(name);
        int width = size > 40 ? 40 : size;
        window.setSize((width+2)*16, 100);
        PixelGrid pGrid = drawImage(image, width, 1);
        window.add(pGrid);
        window.setVisible(true);
        window.repaint();
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    /**
     * Uses the PixelGrid class to display the 2D image read from the
     * MNIST data set
     * @param image - int[][]
     * @param size - the width/height of the image (assumes the image is square)
     */
    public static void displayImage(double[][]image, int size, String name) {
        JFrame window = new JFrame(name);
        window.setSize((size+2)*16, (size+2)*16 + 20);
        PixelGrid pGrid = drawImage(image, size, size);
        window.add(pGrid);
        window.setVisible(true);
        window.repaint();
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    /**
     * Helper method for displayImage that draws the 1D image on the
     * PixelGrid
     * @param image - int[]
     * @param width - int
     * @param height - int
     * @return grid the PixelGrid representation of the image
     */
    public static PixelGrid drawImage(double[] image, int width, int height) {
        PixelGrid grid = new PixelGrid(width, height);
        int c = 0;
        for(int i = 0; i<width; i++) {
            for(int j = 0; j<height; j++) {
                grid.setPixel(image[c++]*255,j,i);
            }
        }
        return grid;
    }

    /**
     * Helper method for displayImage that draws the 2D image on the
     * PixelGrid
     * @param image - int[][]
     * @param width - int
     * @param height - int
     * @return grid the PixelGrid representation of the image
     */
    public static PixelGrid drawImage(double[][] image, int width, int height) {
        PixelGrid grid = new PixelGrid(width, height);
        int c = 0;
        for(int i = 0; i<width; i++) {
            for(int j = 0; j<height; j++) {
                grid.setPixel(image[i][j]*255,j,i);
            }
        }
        return grid;
    }

    /**
     * Initializes the weights in a simple manner to demonstrate functionality
     */
    public void initializeTestWeightsSimple(){
        //Initialize filters to have a center of 1 and C5, F6, and Output weights to 1
        for(int f = 0; f < c1Filters.length; f++) {
            c1Filters[f][2][2] = 1.0;
        }
        for(int f = 0; f < c3Filters.length; f++) {
            c3Filters[f][2][2] = 1.0;
        }
        for(int f = 0; f < c5Filters.length; f++) {
            c5Filters[f][2][2] = 1.0;
        }
        for(int i = 0; i < s2Weights.length; i++){
            s2Weights[i] = 1.0;
        }
        for(int i = 0; i < s4Weights.length; i++){
            s4Weights[i] = 1.0;
        }
        for(int i = 0; i < c5Weights.length; i++){
            c5Weights[i] = 1.0;
        }
        for(int i = 0; i < c5Biases.length; i++){
            c5Biases[i] = 1.0;
        }
        for(int i = 0; i < f6Weights.length; i++){
            for(int j = 0; j < f6Weights[i].length; j++){
                f6Weights[i][j] = 1.0;
            }
        }
        for(int i = 0; i < f6Biases.length; i++){
            f6Biases[i] = 1.0;
        }
        for(int i = 0; i < outputWeights.length; i++){
            for(int j = 0; j < outputWeights[i].length; j++){
                outputWeights[i][j] = 1.0;
            }
        }
    }

}
