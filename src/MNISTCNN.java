import javax.swing.*;
import java.io.*;

public class MNISTCNN {

    /**
     * Main method
     * @param args - String[]
     */
    public static void main(String[] args){
        testCNNetImage();
    }

    /**
     * Trains and tests the LeNet5 CNNet on the MNIST dataset
     */
    public static void testCNNetImage(){
        int sectionLength = 60000; //Programmer determined: Only reads a certain number of numbers to avoid running out of memory
        //Read in the training data
        System.out.println("Parsing data...");
        double[][] trainImages = readImage32x32("MNIST\\train-images-idx1-ubyte", sectionLength, 28*28);
        System.out.println("Images read. Reading labels...");
        double[] trainOutputs = readLabels("MNIST\\train-labels-idx1-ubyte", sectionLength);
        System.out.println("Labels read. Training data parsed.");

        //Train the network
//        LeNet5 net = new LeNet5();
//        net.initNetwork(trainImages, trainOutputs);
//        net.trainNetwork(10);

        //Read in the testing data
        double[][] testImages = readImage32x32("MNIST\\t10k-images-idx3-ubyte", 10000, 28*28);
        //Read the expected outputs
        double[] testOutputs = readLabels("MNIST\\t10k-labels-idx1-ubyte", 10000);

//        //Test the network
//        net.testNetworkImages(testImages, testOutputs);
    }

    /**
     * Reads the MNIST images from the specified filepath. Reads only as many images
     * as specified by sectionLength
     * @param filepath - String
     * @param sectionLength - int
     * @param imageLength - int
     * @return the double[][] of un-padded MNIST images
     */
    public static double[][] readImage(String filepath, int sectionLength, int imageLength){
        //Read the images
        double[][] inputImages = new double[sectionLength][imageLength];
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
            for (int j = 0; j < inputImages.length; j++) {
                for (int i = 0; i < imageLength; i++) {
                    inputImages[j][i] = inputFile.read();
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
     * Reads the MNIST images from the specified filepath and pads them to be 32x32.
     * Reads only as many images as specified by sectionLength
     * @param filepath - String
     * @param sectionLength - int
     * @param imageSize - int
     * @return the double[][] of size 32x32 MNIST images
     */
    public static double[][] readImage32x32(String filepath, int sectionLength, int imageSize){
        //Read the images
        double[][] inputImages = new double[sectionLength][32 * 32];
        int edgeSize = 0;
        if(imageSize != (32*32)){
            edgeSize = (int)(32 - (Math.sqrt(imageSize)));
        }

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
            for (int j = 0; j < inputImages.length; j++) {
                for (int i = 0; i < imageSize + edgeSize; i++) {
                    inputImages[j][i+edgeSize] = inputFile.read();
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
     * Uses the PixelGrid class to display the image read from the
     * MNIST data set
     * @param image - int[]
     */
    public static void displayImage(int[]image) {
        JFrame window = new JFrame("image");
        int width = 28;
        int height = 28;
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
        for(int i = 0; i<28; i++) {
            for(int j = 0; j<28; j++) {
                grid.setPixel(image[c++],j,i);
            }
        }
        return grid;
    }
}
