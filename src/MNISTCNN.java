import javax.swing.*;
import java.io.*;

public class MNISTCNN {

    /**
     * Main method
     * @param args - String[]
     */
    public static void main(String[] args){
//        testCNNetImage();
        LeNet5 net = new LeNet5();
        net.testPooling();
    }

    /**
     * Trains and tests the LeNet5 CNNet on the MNIST dataset
     */
    public static void testCNNetImage(){
        int sectionLength = 60000; //Programmer determined: Only reads a certain number of numbers to avoid running out of memory
        //Read in the training data
        System.out.println("Parsing data...");
        double[][][] trainImages = readImage("MNIST\\train-images-idx3-ubyte", sectionLength, 28, 28);
        System.out.println("Images read. Reading labels...");
        double[] trainOutputs = readLabels("MNIST\\train-labels-idx1-ubyte", sectionLength);
        System.out.println("Labels read. Training data parsed.");

        //Train the network
        System.out.println("Training network...");
//        LeNet5 net = new LeNet5();
//        net.initNetwork(trainImages, trainOutputs);
//        net.trainNetwork(10);

        //Read in the testing data
//        System.out.println("Reading testing data...");
        double[][][] testImages = readImage("MNIST\\t10k-images-idx3-ubyte", 10000, 28, 28);
        //Read the expected outputs
        double[] testOutputs = readLabels("MNIST\\t10k-labels-idx1-ubyte", 10000);

        //Test the network
//        System.out.println("Testing network...");
//        net.testNetworkImages(testImages, testOutputs);
//        System.out.println("Testing complete.");
    }

    /**
     * Reads the MNIST images from the specified filepath. Reads only as many images
     * Creates a 3D array of the images. The first index is the image number
     * as specified by sectionLength
     * @param filepath - String
     * @param sectionLength - int
     * @param imageWidth - int
     * @param imageHeight - int
     * @return the double[][] of un-padded MNIST images
     */
    public static double[][][] readImage(String filepath, int sectionLength, int imageWidth, int imageHeight){
        //Read the images
        double[][][] inputImages = new double[sectionLength][imageWidth][imageHeight];
        File trainingFile = new File(filepath);
        BufferedInputStream inputFile;
        try{
            inputFile = new BufferedInputStream(new FileInputStream(trainingFile));
        }catch(FileNotFoundException e){
            System.err.println(e);
            return null;
        }
        try{
            inputFile.skip(16);
            for (int i = 0; i < inputImages.length; i++) {
                for (int r = 0; r < imageWidth; r++) {
                    for(int c = 0; c < imageHeight; c++) {
                        inputImages[i][r][c] = normalize(inputFile.read());

                    }
                }
            }
            inputFile.close();
            return inputImages;
        }catch(IOException e){
            System.err.println(e);
            return null;
        }
    }

    /**
     * Reads the MNIST labels from the specified filepath. Reads only as many labels as
     * specified by sectionLength
     * @param filepath - String
     * @param sectionLength - int
     * @return the double[] array of labels
     */
    public static double[] readLabels(String filepath, int sectionLength){
        //Read the expected outputs
        double[] desiredOutputs = new double[sectionLength];
        File trainingSolutions = new File(filepath);
        BufferedInputStream solutionFile;
        try{
            solutionFile = new BufferedInputStream(new FileInputStream(trainingSolutions));
        }catch(FileNotFoundException e){
            System.err.println(e);
            return null;
        }
        try{
            solutionFile.skip(8);
            for (int j = 0; j < desiredOutputs.length; j++) {
                desiredOutputs[j] = solutionFile.read();
            }
            solutionFile.close();
            return desiredOutputs;
        }catch(IOException e){
            System.err.println(e);
            return null;
        }
    }

    /**
     * Normalizes the pixel so that the background (white) has a value
     * of -0.1 and the foreground (black) has a value of 1.175 to make
     * training faster
     * @param pixel - double
     * @return the double normalized pixel
     */
    public static double normalize(double pixel){
        return pixel/255.0;
    }

    /**
     * Uses the PixelGrid class to display the image read from the
     * MNIST data set
     * @param image - int[]
     */
    public static void displayImage(int[]image, int size) {
        JFrame window = new JFrame("image");
        int width = size;
        int height = size;
        window.setSize((width+2)*16, (height+2)*16 + 20);
        PixelGrid pGrid = drawImage(image, width, height);
        window.add(pGrid);
        window.setVisible(true);
        window.repaint();
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    /**
     * Helper method for displayImage that draws the image on the
     * PixelGrid
     * @param image - int[]
     * @param width - int
     * @param height - int
     * @return grid the PixelGrid representation of the image
     */
    public static PixelGrid drawImage(int[] image, int width, int height) {
        PixelGrid grid = new PixelGrid(width, height);
        int c = 0;
        for(int i = 0; i<width; i++) {
            for(int j = 0; j<height; j++) {
                grid.setPixel(image[c++],j,i);
            }
        }
        return grid;
    }
}
